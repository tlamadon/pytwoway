'''
Utility functions
'''
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt

def weighted_mean(v, w):
    '''
    Compute weighted mean.

    Arguments:
        v (NumPy Array): vector to weight
        w (NumPy Array): weights

    Returns:
        (NumPy Array): weighted mean
    '''
    if isinstance(w, (float, int)):
        return np.mean(v)
    return np.sum(w * v) / np.sum(w)

def weighted_var(v, w, dof=0):
    '''
    Compute weighted variance.

    Arguments:
        v (NumPy Array): vector to weight
        w (NumPy Array): weights
        dof (int): degrees of freedom

    Returns:
        (NumPy Array): weighted variance
    '''
    m0 = weighted_mean(v, w)

    if isinstance(w, (float, int)):
        n = len(v)
        return np.sum((v - m0) ** 2) / (n - dof)

    return np.sum(w * (v - m0) ** 2) / (np.sum(w) - dof)

def weighted_cov(v1, v2, w1, w2, dof=0):
    '''
    Compute weighted covariance.

    Arguments:
        v1 (NumPy Array): vector to weight
        v2 (NumPy Array): vector to weight
        w1 (NumPy Array): weights for v1
        w2 (NumPy Array): weights for v2
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
    Faster computation of the diagonal of the product of two matrices (i.e. compute (m1 @ m2).diagonal()). Source: comment by hpaulj on https://stackoverflow.com/a/69872249/17333120.

    Arguments:
        m1 (NumPy Array): left matrix
        m2 (NumPy Array): right matrix
    '''
    if m1.shape[1] < m2.shape[0]:
        return diag_of_prod(m2.T, m1.T)
    return np.squeeze(m1[: m2.shape[0], None, :] @ (m2 @ m2.T)[:, :, None])

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
