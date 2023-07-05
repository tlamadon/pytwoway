'''
Plot firm class proportions over time.
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
from matplotlib import pyplot as plt

def plot_firm_classes_over_time(jdata, sdata, subset='all', firm_order=None, xlabel='year', ylabel='class proportions', title='Firm class proportions over time', dpi=None):
    '''
    Plot firm class proportions over time.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for stayers
        subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
        firm_order (NumPy Array or None): sorted firm class order; None keeps the original firm order
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        title (str): plot title
        dpi (float or None): dpi for plot
    '''
    if (not jdata._col_included('t')) or (not sdata._col_included('t')):
        raise ValueError('jdata and sdata must include time data.')

    ## Unpack parameters ##
    nk = jdata.n_clusters()
    weighted = jdata._col_included('w')

    ## Convert to BipartitePandas DataFrame ##
    if subset == 'movers':
        bdf = jdata
    elif subset == 'stayers':
        bdf = sdata
    elif subset == 'all':
        bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        bdf._set_attributes(jdata)
    bdf = bdf.to_long(is_sorted=True, copy=False)
    if isinstance(bdf, bpd.BipartiteLongCollapsed):
        bdf = bdf.uncollapse(is_sorted=True, copy=False)

    ## Plot over time ##
    t_col = bdf.loc[:, 't'].to_numpy()
    all_t = np.unique(t_col)
    class_proportions = np.zeros([len(all_t), nk])
    for t in all_t:
        bdf_t = bdf.loc[t_col == t, :]
        if weighted:
            w_t = bdf_t.loc[:, 'w'].to_numpy()
        else:
            w_t = None
        # Number of observations per firm class per period
        class_proportions[t, :] = np.bincount(bdf_t.loc[:, 'g'], w_t)
        # Normalize to proportions
        class_proportions[t, :] /= class_proportions[t, :].sum()


    if firm_order is not None:
        ## Reorder firms ##
        class_proportions = class_proportions[:, firm_order]

    ## Compute cumulative sum ##
    class_props_cumsum = np.cumsum(class_proportions, axis=1)

    ## Plot ##
    fig, ax = plt.subplots(dpi=dpi)
    x_axis = all_t.astype(str)
    ax.bar(x_axis, class_proportions[:, 0])
    for t in range(1, len(all_t)):
        ax.bar(x_axis, class_proportions[:, t], bottom=class_props_cumsum[:, t - 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()
