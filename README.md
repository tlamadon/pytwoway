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
from pytwoway import twfe_network
tn = twfe_network.twfe_network
# Create twfe object
tw_net = tn.twfe_network(data, formatting, col_dict)
# Convert long data into event study data (not necessary if the data is already in event study format):
tw_net.refactor_es()
# Run the bias-corrected AKM estimator:
tw_net.run_akm_corrected(user_akm)
# Cluster firms based on their wage CDFs (required for the CRE estimator)
tw_net.cluster(user_cluster)
# Run the CRE estimator
tw_net.run_cre(user_cre)
```

- If you want to simulate data
```python
from pytwoway import sim_twfe_network
sn = sim_twfe_network.sim_twfe_network
# Create simulated twfe object
stw_net = sn(sim_params)
# Generate data
sim_data = stw_net.sim_network()
```

- If you want to run Monte Carlo on simulated data
```python
from pytwoway import sim_twfe_network
sn = sim_twfe_network.sim_twfe_network
# Create simulated twfe object
stw_net = sn(sim_params)
# Run Monte Carlo
stw_net.twfe_monte_carlo(N, ncore, akm_params, cre_params, cluster_params)
# Plot results
stw_net.plot_monte_carlo()
```
