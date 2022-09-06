'''
Diagnostic functions.
'''
from paramsdict import ParamsDict
import numpy as np
from matplotlib import pyplot as plt

# Define default parameter dictionary
plot_extendedeventstudy_params = ParamsDict({
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

def plot_extendedeventstudy(adata, periods_pre=2, periods_post=2, params=None):
    '''
    Generate event study plots. If data is not clustered, will plot all transitions in a single figure.

    Arguments:
        adata (BipartiteExtendedEventStudyBase): extended event study or collapsed extended event study format labor data
        periods_pre (int): number of periods before the transition
        periods_post (int): number of periods after the transition
        params (ParamsDict or None): dictionary of parameters for plotting. Run tw.plot_extendedeventstudy_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.plot_extendedeventstudy_params().
    '''
    if params is None:
        params = plot_extendedeventstudy_params()

    n_clusters = adata.n_clusters()

    # Want n_clusters x n_clusters subplots
    fig, axs = plt.subplots(nrows=n_clusters, ncols=n_clusters, sharex=params['sharex'], sharey=params['sharey'])

    # Create lists of the x values and y columns we want
    x_vals = list(np.arange(-periods_pre, 0)) + list(np.arange(1, periods_pre + 1))
    y_cols = adata.col_reference_dict['y']

    # Get y boundaries
    y_min = 1000
    y_max = -1000

    # Generate plots
    if n_clusters is not None:
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                # Keep if previous firm type is i and next firm type is j
                es_plot = adata.loc[(adata.loc[:, f'g{periods_pre}'].to_numpy() == i) & (adata.loc[:, f'g{periods_pre + 1}'].to_numpy() == j), :]
                y = es_plot.loc[:, y_cols].mean(axis=0)
                yerr = es_plot.loc[:, y_cols].std(axis=0) / (len(es_plot) ** 0.5)
                ax.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
                ax.axvline(0, color='orange', zorder=1)
                ax.set_title(f'{i + 1} to {j + 1} (n={len(es_plot)})', y=params['title_height'], fontdict={'fontsize': params['fontsize']})
                ax.grid()
                y_min = min(y_min, ax.get_ylim()[0])
                y_max = max(y_max, ax.get_ylim()[1])
    else:
        # Plot everything
        y = adata.loc[:, y_cols].mean(axis=0)
        yerr = adata.loc[:, y_cols].std(axis=0) / (len(adata) ** 0.5)
        axs.errorbar(x_vals, y, yerr=yerr, ecolor='red', elinewidth=1, zorder=2)
        axs.axvline(0, color='orange', zorder=1)
        axs.set_title(f'All Transitions (n={len(adata)})', y=params['title_height'], fontdict={'fontsize': params['fontsize']})
        axs.grid()
        y_min = min(y_min, axs.get_ylim()[0])
        y_max = max(y_max, axs.get_ylim()[1])

    # Plot
    plt.setp(axs, xticks=np.arange(-periods_pre, periods_post + 1), yticks=np.round(np.linspace(y_min, y_max, 4), params['yticks_round']))
    plt.tight_layout()
    plt.show()
