# pytwoway
Two way fixed effect models for labor in python

Full documentation can be found [here](https://tlamadon.github.io/pytwoway/).

Quick start:

To install from pip:
```shell
pip install pytwoway
```

To run using command line interface:
```shell
pytw --my-config config.txt --akm --cre
```
Example config.txt:
```
data = file.csv
filetype = csv
col_dict = "{'fid': 'your_firmid_col', 'wid': 'your_workerid_col', 'year': 'your_year_col', 'comp': 'your_compensation_col'}"
```

To run in Python:
- If you have data
```python
from pytwoway import twfe_network as tn
# Create twfe object
tw_net = tn.TwoWay(data, formatting, col_dict)
# Prepare data for FE estimator
tw_net.prep_fe()
# Fit FE estimator
tw_net.fit_fe(user_fe)
# Prepare data for CRE estimator
tw_net.prep_cre(user_cluster)
# Run the CRE estimator
tw_net.run_cre(user_cre)
```

- If you want to simulate data
```python
from pytwoway import sim_twfe_network as sn
# Create simulated twfe object
stw_net = sn.SimTwoWay(sim_params)
# Generate data
sim_data = stw_net.sim_network()
```

- If you want to run Monte Carlo on simulated data
```python
from pytwoway import sim_twfe_network as sn
# Create simulated Monte Carlo object
smc_net = sn.TwoWayMonteCarlo(sim_params)
# Run Monte Carlo
smc_net.twfe_monte_carlo(N, ncore, akm_params, cre_params, cluster_params)
# Plot results
smc_net.plot_monte_carlo()
```
