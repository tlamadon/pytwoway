'''
Diagnostic functions.
'''
from paramsdict import ParamsDict
import numpy as np

# Define default parameter dictionary
plot_extended_eventstudy_params = ParamsDict({
    'title_height': (1, 'type', (int, float),
        '''
            (default=1) Location of titles for subfigures.
        ''', None),
    'fontsize': (9, 'type', (int, float),
        '''
            (default=9) Font size of titles for subfigures.
        ''', None),
    'sharex': (True, 'type', bool,
        '''
            (default=True) Share x axis between subfigures.
        ''', None),
    'sharey': (True, 'type', bool,
        '''
            (default=True) Share y axis between subfigures.
        ''', None),
    'yticks_round': (1, 'type', int,
        '''
            (default=1) How many digits to round y ticks.
        ''', None)
})

def plot_extended_eventstudy(adata, transition_col='j', outcomes=['g', 'y'], periods_pre=2, periods_post=2, stable_pre=None, stable_post=None, plot_extended_eventstudy_params=None, is_sorted=False, copy=True):
    '''
    Generate event study plots. If data is not clustered, will plot all transitions in a single figure.

    Arguments:
        transition_col (str): column to use to define a transition
        outcomes (column name or list of column names or None): columns to include data for all periods; None is equivalent to ['g', 'y']
        periods_pre (int): number of periods before the transition
        periods_post (int): number of periods after the transition
        stable_pre (column name or list of column names or None): for each column, keep only workers who have constant values in that column before the transition; None is equivalent to []
        stable_post (column name or list of column names or None): for each column, keep only workers who have constant values in that column after the transition; None is equivalent to []
        plot_extended_eventstudy_params (ParamsDict or None): dictionary of parameters for plotting. Run bpd.plot_extended_eventstudy_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.plot_extended_eventstudy_params().
        is_sorted (bool): if False, dataframe will be sorted by i (and t, if included). Sorting may alter original dataframe if copy is set to False. Set is_sorted to True if dataframe is already sorted.
        copy (bool): if False, avoid copy
    '''
    # FIXME this method raises the following warnings:
    # ResourceWarning: unclosed event loop <_UnixSelectorEventLoop running=False closed=False debug=False> source=self)
    # ResourceWarning: Enable tracemalloc to get the object allocation traceback

    if outcomes is None:
        outcomes = ['g', 'y']

    if plot_extended_eventstudy_params is None:
        plot_extended_eventstudy_params = bpd.plot_extended_eventstudy_params()

    from matplotlib import pyplot as plt

    n_clusters = self.n_clusters()
    added_g = False
    if n_clusters is None:
        # If data not clustered
        added_g = True
        n_clusters = 1
        self.loc[:, 'g'] = 0

    es = self.get_extended_eventstudy(transition_col=transition_col, outcomes=outcomes, periods_pre=periods_pre, periods_post=periods_post, stable_pre=stable_pre, stable_post=stable_post,is_sorted=is_sorted, copy=copy)

    # Want n_clusters x n_clusters subplots
    fig, axs = plt.subplots(nrows=n_clusters, ncols=n_clusters, sharex=plot_extended_eventstudy_params['sharex'], sharey=plot_extended_eventstudy_params['sharey'])
    # Create lists of the x values and y columns we want
    x_vals = []
    y_cols = []
    for i in range(1, periods_pre + 1):
        x_vals.insert(0, - i)
        y_cols.insert(0, f'y_l{i}')
    for i in range(1, periods_post + 1):
        x_vals.append(i)
        y_cols.append(f'y_f{i}')
    # Get y boundaries
    y_min = 1000
    y_max = -1000
    # Generate plots
    if n_clusters > 1:
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                # Keep if previous firm type is i and next firm type is j
                es_plot = es.loc[(es.loc[:, 'g_l1'].to_numpy() == i) & (es.loc[:, 'g_f1'].to_numpy() == j), :]
                y = es_plot.loc[:, y_cols].mean(axis=0)
                yerr = es_plot.loc[:, y_cols].std(axis=0) / (len(es_plot) ** 0.5)
                ax.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
                ax.axvline(0, color='orange', zorder=1)
                ax.set_title(f'{i + 1} to {j + 1} (n={len(es_plot)})', y=plot_extended_eventstudy_params['title_height'], fontdict={'fontsize': plot_extended_eventstudy_params['fontsize']})
                ax.grid()
                y_min = min(y_min, ax.get_ylim()[0])
                y_max = max(y_max, ax.get_ylim()[1])
    else:
        # Plot everything
        y = es.loc[:, y_cols].mean(axis=0)
        yerr = es.loc[:, y_cols].std(axis=0) / (len(es) ** 0.5)
        axs.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
        axs.axvline(0, color='orange', zorder=1)
        axs.set_title(f'All Transitions (n={len(es)})', y=plot_extended_eventstudy_params['title_height'], fontdict={'fontsize': plot_extended_eventstudy_params['fontsize']})
        axs.grid()
        y_min = min(y_min, axs.get_ylim()[0])
        y_max = max(y_max, axs.get_ylim()[1])

    if added_g:
        # Drop g column
        self.drop('g', axis=1, inplace=True)

    # Plot
    plt.setp(axs, xticks=np.arange(-periods_pre, periods_post + 1), yticks=np.round(np.linspace(y_min, y_max, 4), plot_extended_eventstudy_params['yticks_round']))
    plt.tight_layout()
    plt.show()
