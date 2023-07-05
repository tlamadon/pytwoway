'''
Plot earnings asymmetry.
'''
import numpy as np
from matplotlib import pyplot as plt

def plot_earnings_asymmetry(jdata, title='Earnings asymmetry', axis_label='mean log-earnings', circle_scale=1):
    '''
    Generate earnings asymmetry plot.

    Arguments:
        jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
        title (str): plot title
        axis_label (str): label for axes
        circle_scale (float): size scale for circles
    '''
    ## Take subset ##
    # NOTE: workers must change clusters for this plot
    G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)
    G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=True)
    jdata = jdata.loc[(G1 != G2), :]

    ## Unpack parameters ##
    # Number of clusters
    nk = jdata.n_clusters()

    # Clusters
    G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)
    G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=True)
    G12 = G1 + nk * G2
    G21 = G2 + nk * G1

    # Income
    Y1 = jdata.loc[:, 'y1'].to_numpy()
    Y2 = jdata.loc[:, 'y2'].to_numpy()

    # Weights
    weighted = jdata._col_included('w')
    if weighted:
        w1 = jdata.loc[:, 'w1'].to_numpy()
        w2 = jdata.loc[:, 'w2'].to_numpy()
        w = np.sqrt(w1 * w2)
        Y_mean = (w1 * Y1 + w2 * Y2) / (w1 + w2)
    else:
        w = None
        Y_mean = (Y1 + Y2) / 2

    ## Compute cluster-pair mean income ##
    upward_mobility = (G1 < G2)
    downward_mobility = (G1 > G2)
    if weighted:
        w_upward = np.bincount(G12[upward_mobility], w[upward_mobility])
        w_downward = np.bincount(G21[downward_mobility], w[downward_mobility])
    else:
        w_upward = np.bincount(G12[upward_mobility])
        w_downward = np.bincount(G21[downward_mobility])
    Y_upward = np.bincount(G12[upward_mobility], Y_mean[upward_mobility])
    Y_downward = np.bincount(G21[downward_mobility], Y_mean[downward_mobility])

    # Account for missing groups
    w_upward = w_upward[w_upward > 0]
    w_downward = w_downward[w_downward > 0]
    Y_upward = Y_upward[Y_upward != 0]
    Y_downward = Y_downward[Y_downward != 0]

    ## Plot ##
    plt.scatter(Y_upward, Y_downward, s=circle_scale * (w_upward + w_downward), color='black', alpha=0.4)
    # 45 degree line
    min_val = min(Y_upward.min(), Y_downward.min())
    max_val = max(Y_upward.max(), Y_downward.max())
    n_pts = len(Y_upward)
    plt.plot(np.linspace(min_val, max_val, n_pts), np.linspace(min_val, max_val, n_pts), '--', color='black', alpha=0.6)
    plt.xlabel(f'{axis_label}, upward mobility')
    plt.ylabel(f'{axis_label}, downward mobility')
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()
