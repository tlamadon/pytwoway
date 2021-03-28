'''
Utility functions
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def melt(a, col_names):
    '''
    Flatten data and generate columns with corresponding index from multidimensional a.

    Source: https://stackoverflow.com/a/64794686
    https://stackoverflow.com/a/65996547
    '''
    a_indices = np.meshgrid(*(range(a.shape[i]) for i in range(len(a.shape))))
    a_df = pd.DataFrame({col: a_indices[i].ravel() for i, col in enumerate(col_names)})
    a_df['value'] = a.ravel()

    return a_df

# Source for the following 2 functions:
# https://stackoverflow.com/a/21276920
def _rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter_scatter(x, y, **kwargs):
    return plt.scatter(_rand_jitter(x), _rand_jitter(y), **kwargs)
