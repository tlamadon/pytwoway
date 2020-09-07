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