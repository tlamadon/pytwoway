'''
Tests for twfe_network.py

DATE: October 2020
'''
import pytest
import pandas as pd
import pytwoway as tw
import bipartitepandas as bpd

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

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    bdf = bpd.BipartiteLong(df, col_dict=col_dict)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    fe_params = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'h2': False, 'out': 'res_fe.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False, 'Q': 'cov(alpha, psi)', 'data': bdf.get_cs()}

    fe_solver = tw.FEEstimator(fe_params)
    fe_solver.fit_1()
    fe_solver.construct_Q()
    fe_solver.fit_2()

    psi_hat, alpha_hat = fe_solver.get_fe_estimates()

    assert abs(psi_hat[0] - 1) < 1e-5
    assert abs(psi_hat[1] + 1) < 1e-5
    assert abs(alpha_hat[0] - 7) < 1e-5
    assert abs(alpha_hat[1] - 6) < 1e-5
    assert abs(alpha_hat[2] - 8) < 1e-5
