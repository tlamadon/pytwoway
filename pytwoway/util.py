'''
Utility functions
'''

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
