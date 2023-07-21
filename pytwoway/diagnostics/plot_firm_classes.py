'''
Plots related to firm classes.
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
from matplotlib import pyplot as plt

def _plot_worker_types_over_time(bdf, subplot, nk, firm_order=None, subplot_title=''):
    '''
    Generate a subplot for plot_worker_types_over_time().

    Arguments:
        bdf (BipartitePandas DataFrame): long format data
        subplot (MatPlotLib Subplot): subplot
        nk (int): number of firm classes
        firm_order (NumPy Array or None): sorted firm class order; None keeps the original firm order
        subplot_title (str): subplot title
    '''
    weighted = bdf._col_included('w')

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
    x_axis = all_t.astype(str)
    subplot.bar(x_axis, class_proportions[:, 0])
    for k in range(1, nk):
        subplot.bar(x_axis, class_proportions[:, k], bottom=class_props_cumsum[:, k - 1])
    subplot.set_title(subplot_title)

def plot_firm_class_proportions_over_time(jdata, sdata, breakdown_category=None, n_cols=3, category_labels=None, subset='all', firm_order=None, xlabel='year', ylabel='class proportions', title='Firm class proportions over time', subplot_title=''):
    '''
    Plot firm class proportions over time.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for stayers
        breakdown_category (str or None): str specifies a categorical column, where for each group in the specified category, plot firm class proportions over time within that group; if None, plot firm class proportions over time for the entire dataset
        n_cols (int): (if breakdown_category is specified) number of subplot columns
        category_labels (list or None): (if breakdown_category is specified) specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
        firm_order (NumPy Array or None): sorted firm class order; None keeps the original firm order
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        title (str): plot title
        subplot_title (str): (if breakdown_category is specified) subplot title (subplots will be titled `subplot_title` + category, e.g. if `subplot_title`='k=', then subplots will be titled 'k=1', 'k=2', etc., or if `subplot_title`='', then subplots will be titled '1', '2', etc.)
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

    ## Plot ##
    if breakdown_category is None:
        n_rows = 1
        n_cols = 1
    else:
        unique_cat = np.array(sorted(bdf.unique_ids(breakdown_category)))
        if category_labels is None:
            category_labels = unique_cat
        else:
            category_labels = np.array(category_labels)
        if category_labels is not None:
            cat_order = np.argsort(category_labels)
            unique_cat = unique_cat[cat_order]
            category_labels = category_labels[cat_order]
        n_cat = len(unique_cat)
        n_rows = n_cat // n_cols
        if n_rows * n_cols < n_cat:
            # If the bottom column won't be filled
            n_rows += 1

    ## Create subplots ###
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=True)
    if breakdown_category is None:
        _plot_worker_types_over_time(bdf=bdf, subplot=axs, nk=nk, firm_order=firm_order, subplot_title='')
    else:
        n_plots = 0
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                if i * n_cols + j < n_cat:
                    # Keep category i * n_cols + j
                    cat_ij = unique_cat[i * n_cols + j]
                    if category_labels is None:
                        subplot_title_ij = subplot_title + str(unique_cat[i * n_cols + j])
                    else:
                        subplot_title_ij = subplot_title + str(category_labels[i * n_cols + j])
                    _plot_worker_types_over_time(
                        bdf=bdf.loc[bdf.loc[:, breakdown_category].to_numpy() == cat_ij, :],
                        subplot=ax, nk=nk, firm_order=firm_order, subplot_title=subplot_title_ij
                    )
                    n_plots += 1
                else:
                    fig.delaxes(ax)

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_firm_class_proportions_by_category(jdata, sdata, breakdown_category, category_labels=None, subset='all', firm_order=None, xlabel='category', ylabel='class proportions', title='Firm class proportions by category', dpi=None):
    '''
    Plot firm class proportions broken down by the given category.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for stayers
        breakdown_category (str): categorical column, where firm class proportions are plotted for each group within the category
        category_labels (list or None): (if breakdown_category is specified) specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
        firm_order (NumPy Array or None): sorted firm class order; None keeps the original firm order
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        title (str): plot title
        dpi (float or None): dpi for plot
    '''
    ## Unpack parameters ##
    nk = jdata.n_clusters()
    unique_cat = np.array(sorted(jdata.unique_ids(breakdown_category)))
    n_cat = len(unique_cat)
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

    if weighted:
        w = bdf.loc[:, 'w'].to_numpy()
    else:
        w = None

    ## Compute proportions ##
    class_proportions = np.bincount(bdf.loc[:, 'g'].to_numpy() + nk * bdf.loc[:, breakdown_category].to_numpy(), w).reshape((n_cat, nk))

    # Normalize
    class_proportions /= class_proportions.sum(axis=1)[:, None]

    if firm_order is not None:
        ## Reorder firms ##
        class_proportions = class_proportions[:, firm_order]
    if category_labels is not None:
        ## Reorder categories ##
        cat_order = np.argsort(category_labels)
        class_proportions = class_proportions[cat_order, :]

    ## Compute cumulative sum ##
    class_props_cumsum = np.cumsum(class_proportions, axis=1)

    ## Plot ##
    fig, ax = plt.subplots(dpi=dpi)
    if category_labels is None:
        x_axis = unique_cat.astype(str)
    else:
        x_axis = sorted(category_labels)
    ax.bar(x_axis, class_proportions[:, 0])
    for k in range(1, nk):
        ax.bar(x_axis, class_proportions[:, k], bottom=class_props_cumsum[:, k - 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()
