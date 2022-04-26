'''
Tests for pytwoway
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw
from pytwoway import constraints as cons
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

def test_fe_he_8():
    # Test that HE Pii are equivalent when computed using M^{-1} explicitly or computing each observation one at a time using multi-grid solver
    a = bpd.SimBipartite({'num_ind': 1000, 'seed': 1234}).sim_network()
    # Simulate on non-collapsed data
    b = bpd.BipartiteLong(a).clean_data({'connectedness': 'biconnected_observations'})
    fe_solver_b = tw.FEEstimator(b.copy(), {'he': True, 'he_analytical': True, 'seed': 1234})
    fe_solver_b.fit_1()
    fe_solver_b._create_fe_solver()
    fe_solver_b._compute_leverages_Pii()

    fe_solver_c = tw.FEEstimator(b.copy(), {'he': True, 'he_analytical': False, 'ndraw_Pii': 1, 'seed': 1234})
    fe_solver_c.fit_1()
    fe_solver_c._create_fe_solver()
    fe_solver_c._compute_leverages_Pii()

    assert np.sum(np.abs(fe_solver_b.adata['Sii'] - fe_solver_c.adata['Sii'])) < 1e-4

#######################
##### Monte Carlo #####
#######################

def test_fe_cre_1():
    # Use Monte Carlo to test CRE, FE, FE-HO, and FE-HE estimators.
    twmc_net = tw.MonteCarlo()
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

def test_blm_monotonic_1():
    # Test whether BLM likelihoods are monotonic, using default fit.
    rng = np.random.default_rng(1234)
    ## Set parameters ##
    nl = 3
    nk = 4
    n_control = 2
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True
    })
    sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False
    })
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cts_tv_wi_params = tw.sim_continuous_control_params({
        'worker_type_interaction': True
    })
    sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
        'worker_type_interaction': True,
        'stationary_A': True, 'stationary_S': True
    })
    sim_cts_tv_params = tw.sim_continuous_control_params({
        'worker_type_interaction': False
    })
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        'worker_type_interaction': False,
        'stationary_A': True, 'stationary_S': True
    })
    blm_sim_params = tw.sim_params({
        'nl': nl,
        'nk': nk,
        'firm_size': 10,
        'NNm': np.ones(shape=(nk, nk)).astype(int, copy=False),
        'NNs': np.ones(shape=nk).astype(int, copy=False),
        'mmult': 1000, 'smult': 1000,
        'a1_sig': 1, 'a2_sig': 1, 's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {
            'cat_tv_wi_control': sim_cat_tv_wi_params,
            'cat_tnv_wi_control': sim_cat_tnv_wi_params,
            'cat_tv_control': sim_cat_tv_params,
            'cat_tnv_control': sim_cat_tnv_params
        },
        'continuous_controls': {
            'cts_tv_wi_control': sim_cts_tv_wi_params,
            'cts_tnv_wi_control': sim_cts_tnv_wi_params,
            'cts_tv_control': sim_cts_tv_params,
            'cts_tnv_control': sim_cts_tnv_params
        }
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True
    })
    cat_tnv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False
    })
    cat_tnv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    cts_tv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True
    })
    cts_tnv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    cts_tv_params = tw.continuous_control_params({
        'worker_type_interaction': False
    })
    cts_tnv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
    })
    blm_params = tw.blm_params({
        'nl': nl,
        'nk': nk,
        'n_iters_movers': 150,
        'categorical_controls': {
            'cat_tv_wi_control': cat_tv_wi_params,
            'cat_tnv_wi_control': cat_tnv_wi_params,
            'cat_tv_control': cat_tv_params,
            'cat_tnv_control': cat_tnv_params
        },
        'continuous_controls': {
            'cts_tv_wi_control': cts_tv_wi_params,
            'cts_tnv_wi_control': cts_tnv_wi_params,
            'cts_tv_control': cts_tv_params,
            'cts_tnv_control': cts_tnv_params
        }
    })
    ## Simulate data ##
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM model
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Fit BLM model
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.min(np.diff(blm_fit.liks1)[3:]) > 0
    assert np.min(np.diff(blm_fit.liks0)) > 0

# NOTE: this is commented out because it takes so long to run
# def test_blm_monotonic_2():
#     # Test whether BLM likelihoods are monotonic, using constrained-unconstrained fit.
#     rng = np.random.default_rng(1235)
#     ## Set parameters ##
#     nl = 3
#     nk = 4
#     n_control = 2
#     sim_cat_tv_wi_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True
#     })
#     sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True,
#         'stationary_A': True, 'stationary_S': True
#     })
#     sim_cat_tv_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False
#     })
#     sim_cat_tnv_params = tw.sim_categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False,
#         'stationary_A': True, 'stationary_S': True
#     })
#     sim_cts_tv_wi_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': True
#     })
#     sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': True,
#         'stationary_A': True, 'stationary_S': True
#     })
#     sim_cts_tv_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': False
#     })
#     sim_cts_tnv_params = tw.sim_continuous_control_params({
#         'worker_type_interaction': False,
#         'stationary_A': True, 'stationary_S': True
#     })
#     blm_sim_params = tw.sim_params({
#         'nl': nl,
#         'nk': nk,
#         'firm_size': 10,
#         'NNm': np.ones(shape=(nk, nk)).astype(int, copy=False),
#         'NNs': np.ones(shape=nk).astype(int, copy=False),
#         'mmult': 1000, 'smult': 1000,
#         'a1_sig': 1, 'a2_sig': 1, 's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
#         'categorical_controls': {
#             'cat_tv_wi_control': sim_cat_tv_wi_params,
#             'cat_tnv_wi_control': sim_cat_tnv_wi_params,
#             'cat_tv_control': sim_cat_tv_params,
#             'cat_tnv_control': sim_cat_tnv_params
#         },
#         'continuous_controls': {
#             'cts_tv_wi_control': sim_cts_tv_wi_params,
#             'cts_tnv_wi_control': sim_cts_tnv_wi_params,
#             'cts_tv_control': sim_cts_tv_params,
#             'cts_tnv_control': sim_cts_tnv_params
#         }
#     })
#     cat_tv_wi_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True
#     })
#     cat_tnv_wi_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': True,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     cat_tv_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False
#     })
#     cat_tnv_params = tw.categorical_control_params({
#         'n': n_control,
#         'worker_type_interaction': False,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     cts_tv_wi_params = tw.continuous_control_params({
#         'worker_type_interaction': True
#     })
#     cts_tnv_wi_params = tw.continuous_control_params({
#         'worker_type_interaction': True,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     cts_tv_params = tw.continuous_control_params({
#         'worker_type_interaction': False
#     })
#     cts_tnv_params = tw.continuous_control_params({
#         'worker_type_interaction': False,
#         'cons_a': cons.Stationary(), 'cons_a': cons.Stationary()
#     })
#     blm_params = tw.blm_params({
#         'nl': nl,
#         'nk': nk,
#         'n_iters_movers': 150,
#         'categorical_controls': {
#             'cat_tv_wi_control': cat_tv_wi_params,
#             'cat_tnv_wi_control': cat_tnv_wi_params,
#             'cat_tv_control': cat_tv_params,
#             'cat_tnv_control': cat_tnv_params
#         },
#         'continuous_controls': {
#             'cts_tv_wi_control': cts_tv_wi_params,
#             'cts_tnv_wi_control': cts_tnv_wi_params,
#             'cts_tv_control': cts_tv_params,
#             'cts_tnv_control': cts_tnv_params
#         }
#     })
#     ## Simulate data ##
#     blm_true = tw.SimBLM(blm_sim_params)
#     sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
#     jdata, sdata = sim_data['jdata'], sim_data['sdata']
#     jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
#     sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
#     # Initialize BLM estimator
#     blm_fit = tw.BLMEstimator(blm_params)
#     # Fit BLM estimator
#     blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)

#     assert np.min(np.diff(blm_fit.model.liks1)[:83]) > 0
#     assert np.min(np.diff(blm_fit.model.liks0)) > 0

def test_blm_qi():
    # Test whether BLM posterior probabilities are giving the most weight to the correct type.
    rng = np.random.default_rng(1234)
    nl = 3
    nk = 4
    # Define parameter dictionaries
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 1, 's2_low': 0, 's2_high': 1
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'return_qi': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 1, 's2_low': 0, 's2_high': 1
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(sim_data['jdata'])), **sim_data['jdata'])
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1']
    blm_fit.A2 = sim_params['A2']
    blm_fit.S1 = sim_params['S1']
    blm_fit.S2 = sim_params['S2']
    # Estimate qi matrix
    qi_estimate = blm_fit.fit_movers(jdata=jdata)
    max_qi_col = np.argmax(qi_estimate, axis=1)
    n_correct_qi = np.sum(max_qi_col == jdata['l'])

    assert (n_correct_qi / len(max_qi_col)) >= 0.9

def test_blm_start_at_truth_no_controls():
    # Test whether BLM estimator works when starting at truth with no controls.
    rng = np.random.default_rng(1234)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.max(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) < 0.045
    assert np.max(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) < 0.03
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.03
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.045

def test_blm_full_estimation_no_controls():
    # Test whether BLM estimator works for full estimation with no controls.
    rng = np.random.default_rng(1235)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.max(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) < 0.02
    assert np.max(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) < 0.025
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.02
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cat_tv_wi():
    # Test whether BLM estimator works when starting at truth for categorical, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1236)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-2
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.6
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.65
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.6
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.03
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.025

def test_blm_full_estimation_cat_tv_wi():
    # Test whether BLM estimator works for full estimation for categorical, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1237)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_wi_control': sim_cat_tv_wi_params}
    })
    cat_tv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_wi_control': cat_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    adj_order = (1, 0)
    A1_sum_0_sim = (sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 0])[:, adj_order]
    A1_sum_1_sim = (sim_params['A1'].T + sim_params['A1_cat']['cat_tv_wi_control'][:, 1])[:, adj_order]
    A2_sum_0_sim = (sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 0])[:, adj_order]
    A2_sum_1_sim = (sim_params['A2'].T + sim_params['A2_cat']['cat_tv_wi_control'][:, 1])[:, adj_order]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_wi_control'][:, 1]
    S1_sum_0_sim = (np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 0] ** 2))[:, adj_order]
    S1_sum_1_sim = (np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_wi_control'][:, 1] ** 2))[:, adj_order]
    S2_sum_0_sim = (np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 0] ** 2))[:, adj_order]
    S2_sum_1_sim = (np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_wi_control'][:, 1] ** 2))[:, adj_order]
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_wi_control'][:, 1] ** 2)

    sorted_pk1 = np.reshape(np.reshape(sim_params['pk1'], (nk, nk, nl))[:, :, adj_order], (nk * nk, nl))
    sorted_pk0 = sim_params['pk0'][:, adj_order]

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.4
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.65
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.55
    assert np.prod(np.abs((blm_fit.pk1 - sorted_pk1) / sorted_pk1)) ** (1 / sorted_pk1.size) < 1e-2
    assert np.prod(np.abs((blm_fit.pk0 - sorted_pk0) / sorted_pk0)) ** (1 / sorted_pk0.size) < 0.03

def test_blm_start_at_truth_cat_tnv_wi():
    # Test whether BLM estimator works when starting at truth for categorical, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1238)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_wi_control': sim_cat_tnv_wi_params}
    })
    cat_tnv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_wi_control': cat_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 1.2
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.75
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.35
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.02
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.04
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_wi_control'], blm_fit.A2_cat['cat_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_wi_control'], blm_fit.S2_cat['cat_tnv_wi_control']))

def test_blm_full_estimation_cat_tnv_wi():
    # Test whether BLM estimator works for full estimation for categorical, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1239)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_wi_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_wi_control': sim_cat_tnv_wi_params}
    })
    cat_tnv_wi_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_wi_control': cat_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_wi_control'][:, 1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_wi_control'][:, 1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_wi_control'][:, 1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_wi_control'][:, 1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_wi_control'][:, 1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_wi_control'][:, 1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_wi_control'][:, 1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-2
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 1.05
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.55
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.3
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_wi_control'], blm_fit.A2_cat['cat_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_wi_control'], blm_fit.S2_cat['cat_tnv_wi_control']))

def test_blm_start_at_truth_cat_tv():
    # Test whether BLM estimator works when starting at truth for categorical, time-varying control variables.
    rng = np.random.default_rng(1240)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-4
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.6
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.35
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.035

def test_blm_full_estimation_cat_tv():
    # Test whether BLM estimator works for full estimation for categorical, time-varying control variables.
    # NOTE: n_init increased to 40
    rng = np.random.default_rng(1241)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=40, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 0.035
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 1.15
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.65
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cat_tnv():
    # Test whether BLM estimator works when starting at truth for categorical, time non-varying control variables.
    rng = np.random.default_rng(1242)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_control': sim_cat_tnv_params}
    })
    cat_tnv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_control': cat_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-4
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.45
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.45
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_control'], blm_fit.A2_cat['cat_tnv_control'], atol=1e-3))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_control'], blm_fit.S2_cat['cat_tnv_control'], atol=1e-3))

def test_blm_full_estimation_cat_tnv():
    # Test whether BLM estimator works for full estimation for categorical, time non-varying control variables.
    rng = np.random.default_rng(1243)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tnv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tnv_control': sim_cat_tnv_params}
    })
    cat_tnv_params = tw.categorical_control_params({
        'n': n_control,
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tnv_control': cat_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    A1_sum_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][0]
    A1_sum_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_tnv_control'][1]
    A2_sum_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][0]
    A2_sum_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_tnv_control'][1]
    A1_sum_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][0]
    A1_sum_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_tnv_control'][1]
    A2_sum_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][0]
    A2_sum_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_tnv_control'][1]
    S1_sum_0_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][0] ** 2)
    S1_sum_1_sim = np.sqrt(sim_params['S1'].T ** 2 + sim_params['S1_cat']['cat_tnv_control'][1] ** 2)
    S2_sum_0_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][0] ** 2)
    S2_sum_1_sim = np.sqrt(sim_params['S2'].T ** 2 + sim_params['S2_cat']['cat_tnv_control'][1] ** 2)
    S1_sum_0_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][0] ** 2)
    S1_sum_1_fit = np.sqrt(blm_fit.S1.T ** 2 + blm_fit.S1_cat['cat_tnv_control'][1] ** 2)
    S2_sum_0_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][0] ** 2)
    S2_sum_1_fit = np.sqrt(blm_fit.S2.T ** 2 + blm_fit.S2_cat['cat_tnv_control'][1] ** 2)

    assert np.max(np.abs((A1_sum_0_fit - A1_sum_0_sim) / A1_sum_0_sim)) < 0.05
    assert np.max(np.abs((A1_sum_1_fit - A1_sum_1_sim) / A1_sum_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_fit - A2_sum_0_sim) / A2_sum_0_sim)) < 1e-4
    assert np.max(np.abs((A2_sum_1_fit - A2_sum_1_sim) / A2_sum_1_sim)) < 1e-3
    assert np.prod(np.abs((S1_sum_0_fit - S1_sum_0_sim) / S1_sum_0_sim)) ** (1 / S1_sum_0_sim.size) < 0.5
    assert np.prod(np.abs((S1_sum_1_fit - S1_sum_1_sim) / S1_sum_1_sim)) ** (1 / S1_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((S2_sum_0_fit - S2_sum_0_sim) / S2_sum_0_sim)) ** (1 / S2_sum_0_sim.size) < 0.5
    assert np.prod(np.abs((S2_sum_1_fit - S2_sum_1_sim) / S2_sum_1_sim)) ** (1 / S2_sum_1_sim.size) < 0.4
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 1e-2
    assert np.all(np.isclose(blm_fit.A1_cat['cat_tnv_control'], blm_fit.A2_cat['cat_tnv_control'], atol=1e-4))
    assert np.all(np.isclose(blm_fit.S1_cat['cat_tnv_control'], blm_fit.S2_cat['cat_tnv_control'], atol=1e-3))

def test_blm_start_at_truth_cts_tv_wi():
    # Test whether BLM estimator works when starting at truth for continuous, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1244)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_wi_control': sim_cts_tv_wi_params}
    })
    cts_tv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_wi_control': cts_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cts = sim_params['A1_cts'].copy()
    blm_fit.A2_cts = sim_params['A2_cts'].copy()
    blm_fit.S1_cts = sim_params['S1_cts'].copy()
    blm_fit.S2_cts = sim_params['S2_cts'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 0.35
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.55
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_wi_control'] - sim_params['A1_cts']['cts_tv_wi_control']) / sim_params['A1_cts']['cts_tv_wi_control'])) < 1e-2
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_wi_control'] - sim_params['A2_cts']['cts_tv_wi_control']) / sim_params['A2_cts']['cts_tv_wi_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_wi_control'] - sim_params['S1_cts']['cts_tv_wi_control']) / sim_params['S1_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_wi_control'] - sim_params['S2_cts']['cts_tv_wi_control']) / sim_params['S2_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S2_cts']['cts_tv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.03
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.025

def test_blm_full_estimation_cts_tv_wi():
    # Test whether BLM estimator works for full estimation for continuous, time-varying, worker-interaction control variables.
    rng = np.random.default_rng(1245)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_wi_control': sim_cts_tv_wi_params}
    })
    cts_tv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_wi_control': cts_tv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 0.9
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.06
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_wi_control'] - sim_params['A1_cts']['cts_tv_wi_control']) / sim_params['A1_cts']['cts_tv_wi_control'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_wi_control'] - sim_params['A2_cts']['cts_tv_wi_control']) / sim_params['A2_cts']['cts_tv_wi_control'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_wi_control'] - sim_params['S1_cts']['cts_tv_wi_control']) / sim_params['S1_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_wi_control'] - sim_params['S2_cts']['cts_tv_wi_control']) / sim_params['S2_cts']['cts_tv_wi_control'])) ** (1 / sim_params['S2_cts']['cts_tv_wi_control'].size) < 0.9
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 1e-2
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 1e-2

def test_blm_start_at_truth_cts_tnv_wi():
    # Test whether BLM estimator works when starting at truth for continuous, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1246)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_wi_control': sim_cts_tnv_wi_params}
    })
    cts_tnv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_wi_control': cts_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cts = sim_params['A1_cts'].copy()
    blm_fit.A2_cts = sim_params['A2_cts'].copy()
    blm_fit.S1_cts = sim_params['S1_cts'].copy()
    blm_fit.S2_cts = sim_params['S2_cts'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.2
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.5
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_wi_control'] - sim_params['A1_cts']['cts_tnv_wi_control']) / sim_params['A1_cts']['cts_tnv_wi_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_wi_control'] - sim_params['S1_cts']['cts_tnv_wi_control']) / sim_params['S1_cts']['cts_tnv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_wi_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_wi_control'], blm_fit.A2_cts['cts_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_wi_control'], blm_fit.S2_cts['cts_tnv_wi_control']))

def test_blm_full_estimation_cts_tnv_wi():
    # Test whether BLM estimator works for full estimation for continuous, time non-varying, worker-interaction control variables.
    rng = np.random.default_rng(1247)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_wi_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_wi_control': sim_cts_tnv_wi_params}
    })
    cts_tnv_wi_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_wi_control': cts_tnv_wi_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.15
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.95
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_wi_control'] - sim_params['A1_cts']['cts_tnv_wi_control']) / sim_params['A1_cts']['cts_tnv_wi_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_wi_control'] - sim_params['S1_cts']['cts_tnv_wi_control']) / sim_params['S1_cts']['cts_tnv_wi_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_wi_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_wi_control'], blm_fit.A2_cts['cts_tnv_wi_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_wi_control'], blm_fit.S2_cts['cts_tnv_wi_control']))

def test_blm_start_at_truth_cts_tv():
    # Test whether BLM estimator works when starting at truth for continuous, time-varying control variables.
    rng = np.random.default_rng(1248)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_control': sim_cts_tv_params}
    })
    cts_tv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_control': cts_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cts = sim_params['A1_cts'].copy()
    blm_fit.A2_cts = sim_params['A2_cts'].copy()
    blm_fit.S1_cts = sim_params['S1_cts'].copy()
    blm_fit.S2_cts = sim_params['S2_cts'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.35
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 1.7
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_control'] - sim_params['A1_cts']['cts_tv_control']) / sim_params['A1_cts']['cts_tv_control'])) < 1e-5
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_control'] - sim_params['A2_cts']['cts_tv_control']) / sim_params['A2_cts']['cts_tv_control'])) < 1e-5
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_control'] - sim_params['S1_cts']['cts_tv_control']) / sim_params['S1_cts']['cts_tv_control'])) ** (1 / sim_params['S1_cts']['cts_tv_control'].size) < 0.9
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_control'] - sim_params['S2_cts']['cts_tv_control']) / sim_params['S2_cts']['cts_tv_control'])) ** (1 / sim_params['S2_cts']['cts_tv_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 1e-2
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.02

def test_blm_full_estimation_cts_tv():
    # Test whether BLM estimator works for full estimation for continuous, time-varying control variables.
    rng = np.random.default_rng(1249)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tv_params = tw.sim_continuous_control_params({
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tv_control': sim_cts_tv_params}
    })
    cts_tv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': None, 'cons_s': None,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tv_control': cts_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=20, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.6
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.8
    assert np.max(np.abs((blm_fit.A1_cts['cts_tv_control'] - sim_params['A1_cts']['cts_tv_control']) / sim_params['A1_cts']['cts_tv_control'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2_cts['cts_tv_control'] - sim_params['A2_cts']['cts_tv_control']) / sim_params['A2_cts']['cts_tv_control'])) < 1e-5
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tv_control'] - sim_params['S1_cts']['cts_tv_control']) / sim_params['S1_cts']['cts_tv_control'])) ** (1 / sim_params['S1_cts']['cts_tv_control'].size) < 0.9
    assert np.prod(np.abs((blm_fit.S2_cts['cts_tv_control'] - sim_params['S2_cts']['cts_tv_control']) / sim_params['S2_cts']['cts_tv_control'])) ** (1 / sim_params['S2_cts']['cts_tv_control'].size) < 0.85
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.015
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.015

def test_blm_start_at_truth_cts_tnv():
    # Test whether BLM estimator works when starting at truth for continuous, time non-varying control variables.
    rng = np.random.default_rng(1250)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_control': sim_cts_tnv_params}
    })
    cts_tnv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_control': cts_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cts = sim_params['A1_cts'].copy()
    blm_fit.A2_cts = sim_params['A2_cts'].copy()
    blm_fit.S1_cts = sim_params['S1_cts'].copy()
    blm_fit.S2_cts = sim_params['S2_cts'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-3
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 0.75
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 0.75
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_control'] - sim_params['A1_cts']['cts_tnv_control']) / sim_params['A1_cts']['cts_tnv_control'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_control'] - sim_params['S1_cts']['cts_tnv_control']) / sim_params['S1_cts']['cts_tnv_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.035
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 0.025
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_control'], blm_fit.A2_cts['cts_tnv_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_control'], blm_fit.S2_cts['cts_tnv_control']))

def test_blm_full_estimation_cts_tnv():
    # Test whether BLM estimator works for full estimation for continuous, time non-varying control variables.
    # NOTE: n_init increased to 40
    rng = np.random.default_rng(1251)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    # Define parameter dictionaries
    sim_cts_tnv_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'continuous_controls': {'cts_tnv_control': sim_cts_tnv_params}
    })
    cts_tnv_params = tw.continuous_control_params({
        'worker_type_interaction': False,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 0.5, 'a2_sig': 2.5,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'continuous_controls': {'cts_tnv_control': cts_tnv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMEstimator(blm_params)
    # Fit BLM estimator
    blm_fit.fit(jdata=jdata, sdata=sdata, n_init=40, n_best=5, ncore=4, rng=rng)
    blm_fit = blm_fit.model

    assert np.max(np.abs((blm_fit.A1 - sim_params['A1']) / sim_params['A1'])) < 1e-4
    assert np.max(np.abs((blm_fit.A2 - sim_params['A2']) / sim_params['A2'])) < 1e-3
    assert np.prod(np.abs((blm_fit.S1 - sim_params['S1']) / sim_params['S1'])) ** (1 / sim_params['S1'].size) < 1.25
    assert np.prod(np.abs((blm_fit.S2 - sim_params['S2']) / sim_params['S2'])) ** (1 / sim_params['S2'].size) < 1.35
    assert np.max(np.abs((blm_fit.A1_cts['cts_tnv_control'] - sim_params['A1_cts']['cts_tnv_control']) / sim_params['A1_cts']['cts_tnv_control'])) < 1e-4
    assert np.prod(np.abs((blm_fit.S1_cts['cts_tnv_control'] - sim_params['S1_cts']['cts_tnv_control']) / sim_params['S1_cts']['cts_tnv_control'])) ** (1 / sim_params['S1_cts']['cts_tnv_control'].size) < 0.8
    assert np.prod(np.abs((blm_fit.pk1 - sim_params['pk1']) / sim_params['pk1'])) ** (1 / sim_params['pk1'].size) < 0.025
    assert np.prod(np.abs((blm_fit.pk0 - sim_params['pk0']) / sim_params['pk0'])) ** (1 / sim_params['pk0'].size) < 1e-2
    assert np.all(np.isclose(blm_fit.A1_cts['cts_tnv_control'], blm_fit.A2_cts['cts_tnv_control']))
    assert np.all(np.isclose(blm_fit.S1_cts['cts_tnv_control'], blm_fit.S2_cts['cts_tnv_control']))

def test_blm_control_constraints_linear():
    # Test whether Linear() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(1252)
    nl = 3 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Linear(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata, normalize=False)
    blm_fit.fit_stayers(sdata=sdata)

    # Adjust order because worker types are now sorted
    adj_order = (1, 2, 0)

    assert np.max(np.abs(np.diff(np.diff(blm_fit.A1_cat['cat_tv_control'][adj_order, :], axis=0), axis=0))) < 1e-15
    assert np.max(np.abs(np.diff(np.diff(blm_fit.A2_cat['cat_tv_control'][adj_order, :], axis=0), axis=0))) < 1e-15

def test_blm_control_constraints_monotonic():
    # Test whether Monotonic() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(1253)
    nl = 3 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Monotonic(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata, normalize=False)
    blm_fit.fit_stayers(sdata=sdata)

    # Adjust order because worker types are now sorted
    adj_order = (2, 0, 1)

    assert np.min(np.diff(sim_params['A1_cat']['cat_tv_control'], axis=0)) < 0
    assert np.min(np.diff(blm_fit.A1_cat['cat_tv_control'][adj_order, :], axis=0)) >= 0
    assert np.min(np.diff(sim_params['A2_cat']['cat_tv_control'], axis=0)) < 0
    assert np.min(np.diff(blm_fit.A2_cat['cat_tv_control'][adj_order, :], axis=0)) >= 0

def test_blm_control_constraints_stationary_firm_type_variation():
    # Test whether StationaryFirmTypeVariation() constraint for control variables works for BLM estimator.
    rng = np.random.default_rng(1254)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.StationaryFirmTypeVariation(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata, normalize=False)
    blm_fit.fit_stayers(sdata=sdata)

    A1 = blm_fit.A1_cat['cat_tv_control']
    A2 = blm_fit.A2_cat['cat_tv_control']

    assert np.max(np.abs((A2.T - np.mean(A2, axis=1)) - (A1.T - np.mean(A1, axis=1)))) < 1e-14

def test_blm_control_constraints_lb_ub():
    # Test whether BoundedBelow() and BoundedAbove() constraints for control variables work for BLM estimator.
    rng = np.random.default_rng(1255)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_tv_params = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_tv_control': sim_cat_tv_params}
    })
    cat_tv_params = tw.categorical_control_params({
        'n': n_control,
        'cons_a': [cons.BoundedBelow(lb=-0.25), cons.BoundedAbove(ub=0.25)],
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_tv_control': cat_tv_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata, normalize=False)
    blm_fit.fit_stayers(sdata=sdata)

    assert np.min(blm_fit.A1_cat['cat_tv_control']) >= -0.25
    assert np.min(blm_fit.A2_cat['cat_tv_control']) >= -0.25
    assert np.max(blm_fit.A1_cat['cat_tv_control']) <= 0.25
    assert np.max(blm_fit.A2_cat['cat_tv_control']) <= 0.25

def test_blm_control_normalization():
    # Test whether normalization for categorical control variables works for BLM estimator.
    rng = np.random.default_rng(1256)
    nl = 2 # Number of worker types
    nk = 3 # Number of firm types
    n_control = 2 # Number of types for control variable
    # Define parameter dictionaries
    sim_cat_params_one = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': False, 'stationary_S': False,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cat_params_two = tw.sim_categorical_control_params({
        'n': n_control,
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    sim_cts_params = tw.sim_continuous_control_params({
        'stationary_A': True, 'stationary_S': True,
        'worker_type_interaction': True,
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_sim_params = tw.sim_params({
        'nl': nl, 'nk': nk,
        'mmult': 100, 'smult': 100,
        'a1_mu': -2, 'a1_sig': 0.25, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01,
        'categorical_controls': {'cat_control_one': sim_cat_params_one, 'cat_control_two': sim_cat_params_two},
        'continuous_controls': {'cts_control': sim_cts_params}
    })
    cat_params_one = tw.categorical_control_params({
        'n': n_control,
        'cons_a': None, 'cons_s': None,
        'worker_type_interaction': False,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cat_params_two = tw.categorical_control_params({
        'n': n_control,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'worker_type_interaction': True,
        'a1_mu': 0.5, 'a1_sig': 2.5, 'a2_mu': 2, 'a2_sig': 0.25,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    cts_params = tw.continuous_control_params({
        'worker_type_interaction': True,
        'cons_a': cons.Stationary(), 'cons_s': cons.Stationary(),
        'a1_mu': -0.15, 'a1_sig': 0.05, 'a2_mu': 0.15, 'a2_sig': 0.05,
        's1_low': 0, 's1_high': 0.01, 's2_low': 0, 's2_high': 0.01
    })
    blm_params = tw.blm_params({
        'nl': nl, 'nk': nk,
        'a1_mu': -2, 'a1_sig': 0.5, 'a2_mu': 2, 'a2_sig': 0.5,
        's1_low': 0, 's1_high': 0.05, 's2_low': 0, 's2_high': 0.05,
        'categorical_controls': {'cat_control_one': cat_params_one, 'cat_control_two': cat_params_two},
        'continuous_controls': {'cts_control': cts_params}
    })
    # Simulate data
    blm_true = tw.SimBLM(blm_sim_params)
    sim_data, sim_params = blm_true.simulate(return_parameters=True, rng=rng)
    jdata, sdata = sim_data['jdata'], sim_data['sdata']
    jdata = bpd.BipartiteDataFrame(i=np.arange(len(jdata)), **jdata)
    sdata = bpd.BipartiteDataFrame(i=len(jdata) + np.arange(len(sdata)), **sdata)
    # Initialize BLM estimator
    blm_fit = tw.BLMModel(blm_params, rng=rng)
    # Update BLM class attributes to equal truth
    blm_fit.A1 = sim_params['A1'].copy()
    blm_fit.A2 = sim_params['A2'].copy()
    blm_fit.S1 = sim_params['S1'].copy()
    blm_fit.S2 = sim_params['S2'].copy()
    blm_fit.A1_cat = sim_params['A1_cat'].copy()
    blm_fit.A2_cat = sim_params['A2_cat'].copy()
    blm_fit.S1_cat = sim_params['S1_cat'].copy()
    blm_fit.S2_cat = sim_params['S2_cat'].copy()
    blm_fit.A1_cts = sim_params['A1_cts'].copy()
    blm_fit.A2_cts = sim_params['A2_cts'].copy()
    blm_fit.S1_cts = sim_params['S1_cts'].copy()
    blm_fit.S2_cts = sim_params['S2_cts'].copy()
    # Fit BLM estimator
    blm_fit.fit_movers(jdata=jdata)
    blm_fit.fit_stayers(sdata=sdata)

    A1_sum_0_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_0_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][0] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A1_sum_1_0_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 0]
    A1_sum_1_1_sim = sim_params['A1'].T + sim_params['A1_cat']['cat_control_one'][1] + sim_params['A1_cat']['cat_control_two'][:, 1]
    A2_sum_0_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_0_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][0] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A2_sum_1_0_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 0]
    A2_sum_1_1_sim = sim_params['A2'].T + sim_params['A2_cat']['cat_control_one'][1] + sim_params['A2_cat']['cat_control_two'][:, 1]
    A1_sum_0_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_0_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][0] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A1_sum_1_0_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 0]
    A1_sum_1_1_fit = blm_fit.A1.T + blm_fit.A1_cat['cat_control_one'][1] + blm_fit.A1_cat['cat_control_two'][:, 1]
    A2_sum_0_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_0_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][0] + blm_fit.A2_cat['cat_control_two'][:, 1]
    A2_sum_1_0_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 0]
    A2_sum_1_1_fit = blm_fit.A2.T + blm_fit.A2_cat['cat_control_one'][1] + blm_fit.A2_cat['cat_control_two'][:, 1]

    assert np.max(np.abs((A1_sum_0_0_fit - A1_sum_0_0_sim) / A1_sum_0_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_0_1_fit - A1_sum_0_1_sim) / A1_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A1_sum_1_0_fit - A1_sum_1_0_sim) / A1_sum_1_0_sim)) < 1e-2
    assert np.max(np.abs((A1_sum_1_1_fit - A1_sum_1_1_sim) / A1_sum_1_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_0_fit - A2_sum_0_0_sim) / A2_sum_0_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_0_1_fit - A2_sum_0_1_sim) / A2_sum_0_1_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_0_fit - A2_sum_1_0_sim) / A2_sum_1_0_sim)) < 1e-3
    assert np.max(np.abs((A2_sum_1_1_fit - A2_sum_1_1_sim) / A2_sum_1_1_sim)) < 1e-3
    assert np.all(((blm_fit.A1 + blm_fit.A2) / 2)[:, 0] == 0)
    assert np.all(blm_fit.A1_cat['cat_control_two'] == blm_fit.A2_cat['cat_control_two'])
