# pytwoway
Two way fixed effect models in python

To run using default options:
- If you have data
```python
from pytwoway import twfe_network as tn
# data is labor data, formatting is 'long' or 'event study', and col_dict gives a dictionary of column names:
d_net = tn.twfe_network(data, formatting, col_dict)
# Convert long data into event study data (not necessary if the data is already in event study format):
d_net.refactor_es()
# Run the bias-corrected AKM estimator:
d_net.run_akm_corrected()
# Cluster firms based on their wage CDFs (required for the CRE estimator)
d_net.cluster()
# Run the CRE estimator
d_net.run_cre()
```

- If you want to simulate data
```python
from pytwoway import twfe_network as tn
# Parameters are optional when simulating data:
d_net = tn.twfe_network()
# Convert long data into event study data (this is necessary for simulated data):
d_net.refactor_es()
# Run the bias-corrected AKM estimator:
d_net.run_akm_corrected()
# Cluster firms based on their wage CDFs (required for the CRE estimator)
d_net.cluster()
# Run the CRE estimator
d_net.run_cre()
```
