'''
Draw paths of workers to compute covariance between movers
'''
import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
from tqdm.auto import tqdm
from twfe_network import twfe_network as tn

# data = pd.read_feather('../../Google Drive File Stream/.shortcut-targets-by-id/1iN9LApqNxHmVCOV4IUISMwPS7KeZcRhz/ra-adam/data/English/worker_cleaned.ftr')
# col_dict = {'fid': 'codf', 'wid': 'codf_w', 'year': 'year', 'comp': 'comp_current'}
# # d_net for data_network
# d_net = tn(data=data, formatting='long', col_dict=col_dict)
# d_net.clean_data()
# d_net.refactor_es()
# wage_data, wage_cov, wage_sum_cov = compute_path_cov(d_net.data, dir_seq=[1, 1, 1, 0, 0, 0], n_dist=2, n_draws=50, ncore=2)

def valid_neighbors_simple(G, f0, f1):
    '''
    Purpose:
        Find the set of f1's neighbors that are not neighbors of f0

    Inputs:
        G (NetworkX Graph): graph of labor data formatted as event study
        f0 (firm id): firm 0
        f1 (firm id): firm 1

    Returns:
        valid_neighbors (set of firm ids): set of firms that are neighbors of f1 and are not neighbors of f0
    '''
    # Get all neighbors of f0 and f1
    f0_neighbors = set(G.neighbors(f0))
    f1_neighbors = set(G.neighbors(f1))
    # Take the neighbors of f1 that aren't neighbors of f0
    valid_neighbors = f1_neighbors.difference(f0_neighbors)

    return valid_neighbors

def valid_neighbors(G, fids):
    '''
    Purpose:
        Find the set of the last firms neighbors that are sufficiently far from the previous firms (e.g. if there are 3 firms, then we look at neighbors of firm 3 that are at least 2 from firm 2 and at least 3 from firm 1)

    Inputs:
        G (NetworkX Graph): graph of labor data formatted as event study
        fids (list): list of firm ids

    Returns:
        valid_neighbors (set of firm ids): set of firms that are neighbors of the last firm and are sufficiently far from the previous firms
    '''
    if len(fids) == 2:
        return valid_neighbors_simple(G, fids[0], fids[1])
    
    # Get all neighbors of most recent firm
    f_neighbors = set(G.neighbors(fids[-1]))
    # Get firm firm id
    f0 = fids[0]
    # Sufficient distance from f0 is len(fids) (e.g. if 2 firms, want to be at least 2 from f0)
    valid_dist = len(fids)

    # Take the neighbors of most recent firm that are sufficiently far from the first firm
    valid_neighbors = []
    for neighbor in f_neighbors:
        f0_dist = nx.shortest_path_length(G, f0, neighbor)
        if f0_dist == valid_dist:
            valid_neighbors.append(neighbor)

    return valid_neighbors

def draw_worker_path(data_es, G, dir_seq, n_dist=-1):
    '''
    Purpose:
        Draw a path of workers moving into/out of consecutive firms

    Inputs:
        data_es (Pandas DataFrame): labor data formatted as event study
        G (NetworkX Graph): graph of labor data formatted as event study
        dir_seq (list of 0s and 1s): list of movements, where 0 indicates finding a worker who entered a firm while 1 indicates finding a worker who exited a firm
        n_dist (int): number of firms for which to consider a minimum distance of draws, if -1 then ignore distance between firms

    Returns:
        wage_data (list of floats): list of wages along worker path
        move_ids (list of ints): list of move ids along worker path
    '''
    # How many moves to consider
    depth = len(dir_seq)
    # Keep drawing until a valid draw occurs
    valid_draw = False
    while not valid_draw:
        # Create lists of relevant data
        wage_data = []
        move_ids = [] # FIXME
        # Keep track of drawn wids and fids to prevent duplicates
        drawn_wids = []
        drawn_fids = []
        # Draw worker
        worker = data_es.sample(1).iloc[0]
        move_ids.append(worker['move_id']) # FIXME
        drawn_wids.append(worker['wid'])

        if dir_seq[0] == 1:
            wage_data.append(worker['y1'] - worker['y2'])
            drawn_fids.append(worker['f2i'])
            drawn_fids.append(worker['f1i'])
            last_fid = worker['f2i']
        else:
            wage_data.append(worker['y2'] - worker['y1'])
            drawn_fids.append(worker['f1i'])
            drawn_fids.append(worker['f2i'])
            last_fid = worker['f1i']

        for j in range(depth - 1):
            # Draw new workers from the firms where the previous workers moved
            if dir_seq[j + 1] == 1:
                if (j == 0) or (n_dist == -1):
                    # For 2nd draw, no firms blocked
                    df_tmp = data_es[data_es['f1i'] == last_fid]
                else:
                    # For 3rd+ draw, consider only sufficiently far firms, up to n_dist firms away
                    valid_fids = valid_neighbors(G, drawn_fids[-n_dist:])
                    df_tmp = data_es[(data_es['f1i'] == last_fid) & (data_es['f2i'].isin(valid_fids))]
                if len(df_tmp) == 0:
                    # If no observations, just break and valid_draw remains False
                    break
                worker = df_tmp.sample(1).iloc[0]
                wage_data.append(worker['y1'] - worker['y2'])
                last_fid = worker['f2i']
            else:
                if (j == 0) or (n_dist == -1):
                    # For 2nd draw, no firms blocked
                    df_tmp = data_es[data_es['f2i'] == last_fid]
                else:
                    # For 3rd+ draw, consider only sufficiently far firms, up to n_dist firms away
                    valid_fids = valid_neighbors(G, drawn_fids[-n_dist:])
                    df_tmp = data_es[(data_es['f2i'] == last_fid) & (data_es['f1i'].isin(valid_fids))]
                if len(df_tmp) == 0:
                    # If no observations, just break and valid_draw remains False
                    break
                worker = df_tmp.sample(1).iloc[0]
                wage_data.append(worker['y2'] - worker['y1'])
                last_fid = worker['f1i']
            drawn_fids.append(last_fid)
            drawn_wids.append(worker['wid'])
            move_ids.append(worker['move_id']) # FIXME

        # If sufficient draws occurred and no duplicate wids or fids appear, then this is a valid draw
        if (len(drawn_wids) == depth) and (len(drawn_wids) == len(set(drawn_wids))) and (len(drawn_fids) == len(set(drawn_fids))):
            valid_draw = True

    return wage_data, move_ids

def compute_path_cov(data_es, dir_seq, n_dist=-1, n_draws=10000, ncore=1):
    '''
    Purpose:
        Compute covariance of wage changes, going through multiple stages of co-workers moving out of a given firm

    Inputs:
        data_es (Pandas DataFrame): labor data formatted as event study
        dir_seq (list of 0s and 1s): list of movements, where 0 indicates finding a worker who entered a firm while 1 indicates finding a worker who exited a firm
        n_dist (int): number of firms for which to consider a minimum distance of draws, if -1 then ignore distance between firms
        n_draws (int): how many worker paths to sample
        ncore (int): how many cores to use

    Returns:
        wage_data (numpy array): drawn wage data, where rows represent paths and columns represent wage draws along that path
        wage_cov (numpy array): covariance of wage changes
        wage_sum_cov (numpy array): covariance of wages summed across the first and second half of the path
    '''
    # Keep only movers
    data_es = data_es[data_es['m'] == 1]
    # Keep only relevant columns
    data_es = data_es[['f1i', 'f2i', 'y1', 'y2', 'wid']]
    data_es['move_id'] = data_es.index # FIXME

    # Can't compute covariance if only 1 mover, so only keep firms with more than 1 mover
    # Also must ensure any f2i leads back into an f1i, so loop until this condition holds
    prev_len = -1
    new_len = data_es.shape[0]
    while prev_len != new_len:
        prev_len = new_len
        data_es = data_es[data_es.groupby(['f1i'])['y1'].transform('count') > 1]
        data_es = data_es[data_es['f2i'].isin(data_es['f1i'].unique())]
        new_len = data_es.shape[0]

    # To ensure sufficient distance between the firms of movers, we construct a graph of firms
    G = nx.from_pandas_edgelist(data_es, 'f1i', 'f2i')

    # Generate numpy array to hold worker data
    depth = len(dir_seq) # How many moves to consider
    wage_data = np.zeros([n_draws, depth])
    move_ids = np.zeros([n_draws, depth]) # FIXME
    
    if ncore > 1:
        V = []
        # Draw worker paths
        with Pool(processes=ncore) as pool:
            V = pool.starmap( draw_worker_path, [ (data_es, G, dir_seq, n_dist) for _ in range(n_draws)] )
        for i, draws in enumerate(V):
            wage_draw, move_draw = draws
            wage_data[i, :] = wage_draw
            move_ids[i, :] = move_draw
    else:
        for i in tqdm(range(n_draws)):
            wage_draw, move_draw = draw_worker_path(data_es, G, dir_seq, n_dist)
            wage_data[i, :] = wage_draw
            move_ids[i, :] = move_draw

    wage_cov = np.cov(wage_data.T)
    wage_sum_cov = np.cov(np.sum(wage_data[:, : depth // 2], axis=1), np.sum(wage_data[:, depth // 2:], axis=1))

    return wage_data, wage_cov, wage_sum_cov
