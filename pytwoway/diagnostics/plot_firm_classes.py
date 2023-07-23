'''
Plots related to firm classes.
'''
import numpy as np
import pandas as pd
import bipartitepandas as bpd
from pytwoway.util import DxM
from matplotlib import pyplot as plt
import plotly.graph_objects as go

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
    for t_int, t_str in all_t:
        bdf_t = bdf.loc[t_col == t_str, :]
        if weighted:
            w_t = bdf_t.loc[:, 'w'].to_numpy()
        else:
            w_t = None
        # Number of observations per firm class per period
        class_proportions[t_int, :] = np.bincount(bdf_t.loc[:, 'g'], w_t)
        # Normalize to proportions
        class_proportions[t_int, :] /= class_proportions[t_int, :].sum()

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

def plot_firm_class_proportions_over_time(jdata, sdata, breakdown_category=None, n_cols=3, category_labels=None, subset='all', firm_order=None, xlabel='year', ylabel='class proportions', title='Firm class proportions over time', subplot_title='', dpi=None):
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

    ## Plot ##
    if breakdown_category is None:
        n_rows = 1
        n_cols = 1
    else:
        cat_groups = np.array(sorted(bdf.unique_ids(breakdown_category)))
        if category_labels is None:
            category_labels = cat_groups
        else:
            category_labels = np.array(category_labels)
        if category_labels is not None:
            cat_order = np.argsort(category_labels)
            cat_groups = cat_groups[cat_order]
            category_labels = category_labels[cat_order]
        n_cat = len(cat_groups)
        n_rows = n_cat // n_cols
        if n_rows * n_cols < n_cat:
            # If the bottom column won't be filled
            n_rows += 1

    ## Create subplots ##
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=True, dpi=dpi)
    if breakdown_category is None:
        _plot_worker_types_over_time(bdf=bdf, subplot=axs, nk=nk, firm_order=firm_order, subplot_title='')
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        axs.set_title(title)
    else:
        n_plots = 0
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                if i * n_cols + j < n_cat:
                    # Keep category i * n_cols + j
                    cat_ij = cat_groups[i * n_cols + j]
                    if category_labels is None:
                        subplot_title_ij = subplot_title + str(cat_groups[i * n_cols + j])
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
    cat_groups = np.array(sorted(jdata.unique_ids(breakdown_category)))
    n_cat = len(cat_groups)
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
        x_axis = cat_groups.astype(str)
    else:
        x_axis = sorted(category_labels)
    ax.bar(x_axis, class_proportions[:, 0])
    for k in range(1, nk):
        ax.bar(x_axis, class_proportions[:, k], bottom=class_props_cumsum[:, k - 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()

def plot_class_flows(jdata, breakdown_category, method='stacked', category_labels=None, dynamic=False, title='Worker flows', axis_label='category', circle_scale=1, dpi=None, opacity=0.4, font_size=15):
    '''
    Plot flows of workers between each group in a given category.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        breakdown_category (str): categorical column, where worker proportions are plotted for each group within the category
        method (str): 'stacked' for stacked plot; 'sankey' for Sankey plot
        category_labels (list or None): specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        dynamic (bool): if False, plotting estimates from static BLM; if True, plotting estimates from dynamic BLM
        title (str): plot title
        axis_label (str): label for axes (for stacked)
        circle_scale (float): size scale for circles (for stacked)
        dpi (float or None): dpi for plot (for stacked)
        opacity (float): opacity of flows (for Sankey)
        font_size (float): font size for plot (for Sankey)
    '''
    if method not in ['stacked', 'sankey']:
        raise ValueError(f"`method` must be one of 'stacked' or 'sankey', but input specifies {method!r}.")

    ## Extract parameters ##
    cat_groups = np.array(sorted(jdata.unique_ids(breakdown_category)))
    n_cat = len(cat_groups)
    g1 = f'{breakdown_category}1'
    g2 = f'{breakdown_category}'
    if not dynamic:
        g2 += '2'
    else:
        g2 += '4'

    ### Compute NNm ###
    NNm = jdata.groupby(g1)[g2].value_counts().unstack(fill_value=0).to_numpy()
    mover_flows = NNm

    if category_labels is None:
        category_labels = cat_groups + 1
    else:
        ## Sort categories ##
        cat_order = np.argsort(category_labels)
        mover_flows = mover_flows[cat_order, :][:, cat_order]
        category_labels = np.array(category_labels)[cat_order]

    ## Create axes ##
    x_vals, y_vals = np.meshgrid(np.arange(n_cat) + 1, np.arange(n_cat) + 1, indexing='ij')

    if method == 'stacked':
        ## Plot ##
        fig, ax = plt.subplots(dpi=dpi)

        ## Generate plot ##
        ax.scatter(x_vals, y_vals, s=(circle_scale * mover_flows))
        plt.setp(ax, xticks=category_labels, yticks=category_labels)
        ax.set_xlabel(f'{axis_label}, period 1')
        ax.set_ylabel(f'{axis_label}, period 2')
        ax.set_title(title)
        ax.grid()
        plt.show()
    elif method == 'sankey':
        colors = np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207],
                [255, 0, 255]
            ]
        )

        ## Sankey ##
        sankey = go.Sankey(
            # Define nodes
            node=dict(
                pad=15,
                thickness=1,
                line=dict(color='white', width=0),
                label=[f'{axis_label}={category_label}' for category_label in category_labels] + [f'{axis_label}={category_label}' for category_label in category_labels],
                color='white'
            ),
            link=dict(
                # Source firm
                source=(x_vals - 1).flatten(),
                # Destination firm
                target=(y_vals + n_cat - 1).flatten(),
                # Worker flows
                value=mover_flows.flatten(),
                # Color
                color=[f'rgba({str(list(colors[i, :]))[1: -1]}, {opacity})' for i in range(n_cat) for _ in range(n_cat)]
            )
        )

        fig = go.Figure(data=sankey)
        fig.update_layout(title_text=title, font_size=font_size)
        fig.show()
