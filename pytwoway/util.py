'''
Utility functions
'''
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt

def fast_prod_diag_sp(diag, sp):
    '''
    Faster product of diagonal and sparse matrices, i.e. take diag @ sp. Source: https://stackoverflow.com/a/16046783/17333120.

    Arguments:
        diag (NumPy Array): diagonal entries
        sp (CSC Matrix): sparse matrix

    Returns:
        (CSC Matrix): product of diagonal and sparse matrices
    '''
    data = np.take(diag, sp.indices) * sp.data
    ret = csc_matrix((data, sp.indices, sp.indptr), shape=sp.shape)

    return ret

def scramble(lst):
    '''
    Reorder a list from [a, b, c, d, e] to [a, e, b, d, c]. This is used for attrition with multiprocessing, to ensure memory usage stays relatively constant, by mixing together large and small draws. Scrambled lists can be unscrambled with _unscramble().

    Arguments:
        lst (list): list to scramble

    Returns:
        (list): scrambled list
    '''
    new_lst = []
    for i in range(len(lst)):
        if i % 2 == 0:
            new_lst.append(lst[i // 2])
        else:
            new_lst.append(lst[len(lst) - i // 2 - 1])

    return new_lst

def unscramble(lst):
    '''
    Reorder a list from [a, e, b, d, c] to [a, b, c, d, e]. This undoes the scrambling done by _scramble().

    Arguments:
        lst (list): list to unscramble

    Returns:
        (list): unscrambled list
    '''
    front_lst = []
    back_lst = []
    for i, element in enumerate(lst):
        if i % 2 == 0:
            front_lst.append(element)
        else:
            back_lst.append(element)

    return front_lst + list(reversed(back_lst))

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
