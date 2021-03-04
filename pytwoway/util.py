'''
Utility functions
'''
import numpy as np
import pandas as pd

def update_dict(default_params, user_params):
    '''
    Replace entries in default_params with values in user_params. This function allows user_params to include only a subset of the required parameters in the dictionary.

    Arguments:
        default_params (dict): default parameter values
        user_params (dict): user selected parameter values

    Returns:
        params (dict): default_params updated with parameter values in user_params
    '''
    params = default_params.copy()

    params.update(user_params)

    return params

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
