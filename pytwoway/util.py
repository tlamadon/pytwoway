'''
Utility functions
'''
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt

def weighted_mean(v, w=1):
    '''
    Compute weighted mean.

    Arguments:
        v (NumPy Array): vector to weight
        w (NumPy Array or float): weights

    Returns:
        (NumPy Array): weighted mean
    '''
    if isinstance(w, (float, int)):
        return np.mean(v)
    return np.sum(w * v) / np.sum(w)

def weighted_var(v, w=1, dof=0):
    '''
    Compute weighted variance.

    Arguments:
        v (NumPy Array): vector to weight
        w (NumPy Array or float): weights
        dof (int): degrees of freedom

    Returns:
        (NumPy Array): weighted variance
    '''
    m0 = weighted_mean(v, w)

    if isinstance(w, (float, int)):
        n = len(v)
        return np.sum((v - m0) ** 2) / (n - dof)

    return np.sum(w * (v - m0) ** 2) / (np.sum(w) - dof)

def weighted_cov(v1, v2, w1=1, w2=1, dof=0):
    '''
    Compute weighted covariance.

    Arguments:
        v1 (NumPy Array): vector to weight
        v2 (NumPy Array): vector to weight
        w1 (NumPy Array or float): weights for v1
        w2 (NumPy Array or float): weights for v2
        dof (int): degrees of freedom

    Returns:
        (NumPy Array): weighted covariance
    '''
    m1 = weighted_mean(v1, w1)
    m2 = weighted_mean(v2, w2)

    if isinstance(w1, (float, int)) and isinstance(w2, (float, int)):
        n = len(v1)
        return np.sum((v1 - m1) * (v2 - m2)) / (n - dof)

    w3 = np.sqrt(w1 * w2)

    return np.sum(w3 * (v1 - m1) * (v2 - m2)) / (np.sum(w3) - dof)

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    '''
    Very close to numpy.percentile, but supports weights. NOTE: quantiles should be in [0, 1]!

    Arguments:
        values (NumPy Array): data
        quantiles (NumPy Array): quantiles to compute
        sample_weight (NumPy Array): weights
        values_sorted (bool): if True, skips sorting of initial array
        old_style (bool): if True, changes output to be consistent with numpy.percentile

    Returns:
        (NumPy Array): computed quantiles
    '''
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    if not (np.all(quantiles >= 0) and np.all(quantiles <= 1)):
        raise ValueError('Quantiles should be in [0, 1].')

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)

def DxSP(diag, sp):
    '''
    Faster product of a diagonal and a sparse matrix, i.e. take diag @ sp. Source: https://stackoverflow.com/a/16046783/17333120.

    Arguments:
        diag (NumPy Array or float): diagonal entries or multiplicative factor
        sp (CSC Matrix): sparse matrix

    Returns:
        (CSC Matrix): product of diagonal and sparse matrix
    '''
    if isinstance(diag, (float, int)):
        # If multiplicative factor
        if diag == 1:
            return sp
        return diag * sp

    # If diagonal entries
    data = np.take(diag, sp.indices) * sp.data
    ret = csc_matrix((data, sp.indices, sp.indptr), shape=sp.shape)

    return ret

def SPxD(sp, diag):
    '''
    Faster product of a sparse matrix and a diagonal, i.e. take sp @ diag.

    Arguments:
        sp (CSC Matrix): sparse matrix
        diag (NumPy Array or float): diagonal entries or multiplicative factor

    Returns:
        (CSC Matrix): product of sparse matrix and diagonal
    '''
    if isinstance(diag, (float, int)):
        # If multiplicative factor
        if diag == 1:
            return sp
        return diag * sp

    # If diagonal entries
    return sp.multiply(diag)

def DxM(diag, m):
    '''
    Product of a diagonal and a matrix, i.e. take diag @ m.

    Arguments:
        diag (NumPy Array or float): diagonal entries or multiplicative factor
        m (NumPy Array): matrix
    '''
    if isinstance(diag, (float, int)):
        # If multiplicative factor
        if diag == 1:
            return m
        return diag * m

    # If diagonal entries
    return (diag * m.T).T

def MxD(m, diag):
    '''
    Product of a matrix and a diagonal, i.e. take m @ diag.

    Arguments:
        m (NumPy Array): matrix
        diag (NumPy Array or float): diagonal entries or multiplicative factor
    '''
    if isinstance(diag, (float, int)):
        # If multiplicative factor
        if diag == 1:
            return m
        return diag * m

    # If diagonal entries
    return m * diag

def diag_of_sp_prod(m1, m2):
    '''
    Faster computation of the diagonal of the product of two sparse matrices (i.e. compute (m1 @ m2).diagonal()). Sources: https://stackoverflow.com/a/14759273/17333120 and https://stackoverflow.com/a/69872249/17333120.

    Arguments:
        m1 (NumPy Array): left matrix
        m2 (NumPy Array): right matrix
    '''
    if m1.shape[1] < m2.shape[0]:
        return diag_of_sp_prod(m2.T, m1.T)
    return np.asarray(m1[:, : m2.shape[0]].multiply(m2.T).sum(axis=1))[:, 0]

def diag_of_prod(m1, m2):
    '''
    Faster computation of the diagonal of the product of two matrices (i.e. compute (m1 @ m2).diagonal()). Sources: https://stackoverflow.com/a/14759273/17333120 and https://stackoverflow.com/a/69872249/17333120.

    Arguments:
        m1 (NumPy Array): left matrix
        m2 (NumPy Array): right matrix
    '''
    if m1.shape[1] < m2.shape[0]:
        return diag_of_prod(m2.T, m1.T)
    return np.multiply(m1[:, : m2.shape[0]], m2.T).sum(axis=1)

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
