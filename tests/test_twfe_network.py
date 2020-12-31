'''
Tests for twfe_network.py

DATE: October 2020
'''
import pytest
import networkx as nx
import pandas as pd

import os
# Navigate to parent folder for import
#os.chdir('..')

from pytwoway import bipartite_network as bn
from pytwoway import twfe_network as tn
from pytwoway import fe

def test_twfe_refactor_1():
    # Continuous time, 2 movers between firms 1 and 2, and 1 stayer at firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 1., 'index': 1})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 1, 'time': 2, 'id': 2, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 3, 'time': 2, 'id': 3, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.shape[0] == 0

def test_twfe_refactor_2():
    # Discontinuous time, 2 movers between firms 1 and 2, and 1 stayer at firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 2, 'time': 3, 'id': 1, 'comp': 1., 'index': 1})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 1, 'time': 2, 'id': 2, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 3, 'time': 2, 'id': 3, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

def test_twfe_refactor_3():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 1., 'index': 1})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 3
    assert movers.iloc[1]['wid'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 3
    assert movers.iloc[2]['f2i'] == 2
    assert movers.iloc[2]['wid'] == 3
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 2

def test_twfe_refactor_4():
    # Continuous time, 1 mover between firms 1 and 2 and then 2 and 1, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2, 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 1, 'index': 1})
    worker_data.append({'firm': 1, 'time': 3, 'id': 1, 'comp': 1, 'index': 2})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1, 'index': 3})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 1, 'index': 4})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1, 'index': 5})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2, 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 2
    assert movers.iloc[2]['f2i'] == 3
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 3
    assert movers.iloc[3]['f2i'] == 2
    assert movers.iloc[3]['wid'] == 3
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_twfe_refactor_5():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1, 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 1., 'index': 1})
    worker_data.append({'firm': 1, 'time': 4, 'id': 1, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 5})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2., 'index': 6})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 2
    assert movers.iloc[2]['f2i'] == 3
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 3
    assert movers.iloc[3]['f2i'] == 2
    assert movers.iloc[3]['wid'] == 3
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_twfe_refactor_6():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are continuous), 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 1, 'time': 2, 'id': 1, 'comp': 2., 'index': 1})
    worker_data.append({'firm': 2, 'time': 3, 'id': 1, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 1, 'time': 5, 'id': 1, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 1., 'index': 5})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 6})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 2
    assert movers.iloc[2]['f2i'] == 3
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 3
    assert movers.iloc[3]['f2i'] == 2
    assert movers.iloc[3]['wid'] == 3
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_twfe_refactor_7():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 2 and 3, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 1, 'time': 3, 'id': 1, 'comp': 2., 'index': 1})
    worker_data.append({'firm': 2, 'time': 4, 'id': 1, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 1, 'time': 6, 'id': 1, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 1., 'index': 5})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 6})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 2
    assert movers.iloc[2]['f2i'] == 3
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 3
    assert movers.iloc[3]['f2i'] == 2
    assert movers.iloc[3]['wid'] == 3
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_twfe_refactor_8():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 1 and 2, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 1, 'time': 3, 'id': 1, 'comp': 2., 'index': 1})
    worker_data.append({'firm': 2, 'time': 4, 'id': 1, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 1, 'time': 6, 'id': 1, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 1, 'time': 1, 'id': 2, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 2, 'time': 2, 'id': 2, 'comp': 1., 'index': 5})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 6})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 1
    assert movers.iloc[2]['f2i'] == 2
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 3
    assert movers.iloc[3]['f2i'] == 2
    assert movers.iloc[3]['wid'] == 3
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_twfe_refactor_9():
    # Discontinuous time, 1 mover between firms 1 and 2 and then 2 and 1 (but who has 2 periods at firm 1 that are discontinuous), 1 between firms 2 and 1, and 1 between firms 3 and 2, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 1, 'time': 3, 'id': 1, 'comp': 2., 'index': 1})
    worker_data.append({'firm': 2, 'time': 4, 'id': 1, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 1, 'time': 6, 'id': 1, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 1, 'time': 2, 'id': 2, 'comp': 1., 'index': 5})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 6})
    worker_data.append({'firm': 2, 'time': 2, 'id': 3, 'comp': 2., 'index': 7})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 1
    assert movers.iloc[1]['wid'] == 1
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert movers.iloc[2]['f1i'] == 2
    assert movers.iloc[2]['f2i'] == 1
    assert movers.iloc[2]['wid'] == 2
    assert movers.iloc[2]['y1'] == 1
    assert movers.iloc[2]['y2'] == 1

    assert movers.iloc[3]['f1i'] == 3
    assert movers.iloc[3]['f2i'] == 2
    assert movers.iloc[3]['wid'] == 3
    assert movers.iloc[3]['y1'] == 1
    assert movers.iloc[3]['y2'] == 2

def test_twfe_refactor_10():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 3, and 1 stayer at firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 1., 'index': 1})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 3, 'time': 1, 'id': 3, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 3, 'time': 2, 'id': 3, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 3
    assert movers.iloc[1]['wid'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['f1i'] == 3
    assert stayers.iloc[0]['f2i'] == 3
    assert stayers.iloc[0]['wid'] == 3
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_twfe_refactor_11():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 2., 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 1., 'index': 1})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 1., 'index': 2})
    worker_data.append({'firm': 4, 'time': 2, 'id': 2, 'comp': 1., 'index': 3})
    worker_data.append({'firm': 4, 'time': 1, 'id': 3, 'comp': 1., 'index': 4})
    worker_data.append({'firm': 4, 'time': 2, 'id': 3, 'comp': 1., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    df_ES = b_net.data

    stayers = df_ES[df_ES['m'] == 0]
    movers = df_ES[df_ES['m'] == 1]

    assert movers.iloc[0]['f1i'] == 1
    assert movers.iloc[0]['f2i'] == 2
    assert movers.iloc[0]['wid'] == 1
    assert movers.iloc[0]['y1'] == 2
    assert movers.iloc[0]['y2'] == 1

    assert movers.iloc[1]['f1i'] == 2
    assert movers.iloc[1]['f2i'] == 3
    assert movers.iloc[1]['wid'] == 2
    assert movers.iloc[1]['y1'] == 1
    assert movers.iloc[1]['y2'] == 1

    assert stayers.iloc[0]['f1i'] == 3
    assert stayers.iloc[0]['f2i'] == 3
    assert stayers.iloc[0]['wid'] == 3
    assert stayers.iloc[0]['y1'] == 1
    assert stayers.iloc[0]['y2'] == 1

def test_fe_ho_1():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    # psi1 = 5, psi2 = 3, psi4 = 4
    # alpha1 = 3, alpha2 = 2, alpha3 = 4
    worker_data = []
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 8., 'index': 0})
    worker_data.append({'firm': 2, 'time': 2, 'id': 1, 'comp': 6., 'index': 1})
    worker_data.append({'firm': 2, 'time': 1, 'id': 2, 'comp': 5., 'index': 2})
    worker_data.append({'firm': 4, 'time': 2, 'id': 2, 'comp': 6., 'index': 3})
    worker_data.append({'firm': 4, 'time': 1, 'id': 3, 'comp': 8., 'index': 4})
    worker_data.append({'firm': 4, 'time': 2, 'id': 3, 'comp': 8., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'fid': 'firm', 'wid': 'id', 'year': 'time', 'comp': 'comp'}

    b_net = bn.BipartiteData(data=df, col_dict=col_dict)
    b_net.clean_data()
    b_net.refactor_es()

    fe_params = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'hetero': False, 'out': 'res_fe.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False, 'Q': 'cov(alpha, psi)', 'data': b_net}

    fe_solver = fe.FESolver(fe_params)
    fe_solver.fit_1()
    fe_solver.construct_Q()
    fe_solver.fit_2()

    psi_hat, alpha_hat = fe_solver.get_fe_estimates()

    assert abs(psi_hat[1] - 1) < 1e-5
    assert abs(psi_hat[2] + 1) < 1e-5
    assert abs(alpha_hat[1] - 7) < 1e-5
    assert abs(alpha_hat[2] - 6) < 1e-5
    assert abs(alpha_hat[3] - 8) < 1e-5
