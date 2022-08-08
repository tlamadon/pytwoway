'''
Class for simulating from the Borovickova-Shimer dgp.
'''
import numpy as np
from bipartitepandas import BipartiteDataFrame, clean_params
from bipartitepandas.util import ParamsDict, _sort_cols

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq2(a):
    return a >= 2
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _in_minus_1_1(a):
    return -1 <= a <= 1

# Define default parameter dictionaries
_sim_bs_params_default = ParamsDict({
    'n_workers': (1000, 'type_constrained', (int, _gteq1),
        '''
            (default=1000) Number of workers.
        ''', '>= 1'),
    'n_firms': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Number of firms.
        ''', '>= 1'),
    'mean_worker_matches': (2, 'type_constrained', ((float, int), _gteq2),
        '''
            (default=2) Average number of firms each worker works for, drawn from a Poisson distribution. Workers are constrained to work for at least 2 firms.
        ''', '>= 2'),
    'mean_firm_employees': (20, 'type_constrained', ((float, int), _gteq2),
        '''
            (default=20) Average number of employees at each firm, drawn from a Poisson distribution. Firms are constrained to have at least 2 employees.
        ''', '>= 2'),
    'sigma_lambda_sq': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Variance of lambda_i.
        ''', '>= 0'),
    'sigma_mu_sq': (1, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1) Variance of mu_j.
        ''', '>= 0'),
    'sigma_wages': (2, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=2) Standard error of wages. Must be at least sqrt((sigma_lambda ** 2 + sigma_mu ** 2 - 2 * rho * sigma_lambda * sigma_mu) / (1 - rho ** 2)).
        ''', '>= 0'),
    'rho': (0.25, 'type_constrained', ((float, int), _in_minus_1_1),
        '''
            (default=0.25) Correlation between lambda_i and mu_j.
        ''', 'in [-1, 1]')
})

def sim_bs_params(update_dict=None):
    '''
    Dictionary of default sim_bs_params. Run tw.sim_bs_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of sim_bs_params
    '''
    new_dict = _sim_bs_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

class SimBS:
    '''
    Class of SimBS, where SimBS simulates a bipartite Borovickova-Shimer network of firms and workers.

    Arguments:
        sim_params (ParamsDict): dictionary of parameters for simulating data. Run tw.sim_bs_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.sim_bs_params().
    '''

    def __init__(self, sim_params=None):
        if sim_params is None:
            sim_params = sim_bs_params()

        # Check that sigma_wages is large enough
        sigma_lambda_sq, sigma_mu_sq, sigma_wages, rho = sim_params.get_multiple(('sigma_lambda_sq', 'sigma_mu_sq', 'sigma_wages', 'rho'))
        sigma_lambda, sigma_mu = np.sqrt(sigma_lambda_sq), np.sqrt(sigma_mu_sq)

        thres = np.sqrt((sigma_lambda ** 2 + sigma_mu ** 2 - 2 * rho * sigma_lambda * sigma_mu) / (1 - rho ** 2))

        if sigma_wages < thres:
            raise ValueError(f"'sigma_wages' ({sigma_wages:2.2f}) must be at least sqrt(('sigma_lambda' ** 2 + 'sigma_mu' ** 2 - 2 * 'rho' * 'sigma_lambda' * 'sigma_mu') / (1 - 'rho' ** 2)) ({thres:2.2f}).")

        # Store parameters
        self.params = sim_params

    def _simulate_workers_firms(self, rng=None):
        '''
        Simulate number of jobs for each worker and number of employees for each firm to use for simulating bipartite Borovickova-Shimer data.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (tuple): (number of jobs for each worker, number of employees for each firm)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        n_workers, n_firms = self.params.get_multiple(('n_workers', 'n_firms'))
        worker_mean, firm_mean = self.params.get_multiple(('mean_worker_matches', 'mean_firm_employees'))

        # Simulate number of jobs for each worker and number of employees for each firm
        n_matches_workers = np.maximum(rng.poisson(lam=worker_mean, size=n_workers), 2)
        n_matches_firms = np.maximum(rng.poisson(lam=firm_mean, size=n_firms), 2)

        # Ensure sum_workers = sum_firms
        sum_workers = sum(n_matches_workers)
        sum_firms = sum(n_matches_firms)
        while sum_workers != sum_firms:
            if sum_workers > sum_firms:
                # Simulate new firm
                new_firm = max(rng.poisson(lam=firm_mean), 2)
                n_matches_firms = np.concatenate([n_matches_firms, [new_firm]])
            elif sum_firms > sum_workers:
                # Simulate new worker
                new_worker = max(rng.poisson(lam=worker_mean), 2)
                n_matches_workers = np.concatenate([n_matches_workers, [new_worker]])
            sum_workers = sum(n_matches_workers)
            sum_firms = sum(n_matches_firms)

        return (n_matches_workers, n_matches_firms)

    def simulate(self, rng=None):
        '''
        Simulate data. Columns are as follows: i=worker id; j=firm id; y=wage; t=time; lambda_i=lambda_i; mu_j=mu_j.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (BipartiteDataFrame): simulated data

        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        sigma_lambda_sq, sigma_mu_sq, sigma_wages, rho = self.params.get_multiple(('sigma_lambda_sq', 'sigma_mu_sq', 'sigma_wages', 'rho'))
        sigma_lambda, sigma_mu = np.sqrt(sigma_lambda_sq), np.sqrt(sigma_mu_sq)

        # Simulate number of jobs for each worker and number of employees for each firm
        n_matches_workers, n_matches_firms = self._simulate_workers_firms(rng)
        n_workers, n_firms = len(n_matches_workers), len(n_matches_firms)

        # Simulate lambda_i and mu_j
        lambda_i = rng.normal(loc=0, scale=sigma_lambda, size=n_workers)
        mu_j = rng.normal(loc=0, scale=sigma_mu, size=n_firms)

        # Sort firms by mu_j
        firm_order = np.argsort(mu_j)
        n_matches_firms = n_matches_firms[firm_order]
        mu_j = mu_j[firm_order]

        # Match workers and firms
        chi_i_j = rng.normal(loc=np.repeat(lambda_i * rho * sigma_mu / sigma_lambda, n_matches_workers), scale=sigma_mu * np.sqrt(1 - rho ** 2))
        chi_i_j_sorted_idx = np.argsort(chi_i_j)

        # Generate i and j
        i = np.repeat(np.arange(n_workers), n_matches_workers)
        j = np.zeros(len(i), dtype=int)
        j_idx = 0
        for firm, n_matches_firm in enumerate(n_matches_firms):
            j[chi_i_j_sorted_idx[j_idx: j_idx + n_matches_firm]] = firm
            j_idx += n_matches_firm

        # Simulate wages
        a = (sigma_lambda - rho * sigma_mu) / (sigma_lambda * (1 - rho ** 2))
        b = (sigma_mu - rho * sigma_lambda) / (sigma_mu * (1 - rho ** 2))
        sigma_eps = np.sqrt(sigma_wages ** 2 - (sigma_lambda ** 2 + sigma_mu ** 2 - 2 * rho * sigma_lambda * sigma_mu) / (1 - rho ** 2))
        y = a * lambda_i[i] + b * mu_j[j] + rng.normal(loc=0, scale=sigma_eps, size=len(i))

        # Convert into BipartiteDataFrame
        bdf = BipartiteDataFrame(i=i, j=j, y=y, lambda_i=a * lambda_i[i], mu_j=b * mu_j[j]).construct_artificial_time(time_per_worker=True, is_sorted=True, copy=False)

        ## Clean ##
        # Clean parameters
        cp1 = clean_params({'drop_returns': 'returns', 'is_sorted': True, 'copy': False, 'verbose': False})
        cp2 = clean_params({'is_sorted': True, 'copy': False, 'verbose': False})

        # Clean
        bdf = bdf.clean(cp1).min_joint_obs_frame(is_sorted=True, copy=False).clean(cp2)

        # Drop m column
        bdf = bdf.drop('m', axis=1, inplace=True, allow_optional=True)

        return bdf
