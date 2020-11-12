'''
Test code - import modules, simulate data, and test
'''
import pandas as pd
import twfe_network as tn
import fe_approximate_correction_full as fe

##### Italian #####
data = pd.read_feather('../../Google Drive File Stream/.shortcut-targets-by-id/1iN9LApqNxHmVCOV4IUISMwPS7KeZcRhz/ra-adam/data/English/worker_cleaned.ftr')
col_dict = {'fid': 'codf', 'wid': 'codf_w', 'year': 'year', 'comp': 'comp_current'}
net = tn.twfe_network(data, col_dict=col_dict)
##### Italian #####

##### Simulated #####
# import sim_twfe_network as sim
# a = sim.sim_twfe_network()
# data = a.sim_network()
# net = tn.twfe_network(data)
##### Simulated #####

net.clean_data()
net.refactor_es()
net.cluster()

akm_params = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'hetero': False, 'out': 'res_akm.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False, 'Q': 'cov(psi_i, psi_j)', 'data': net.data}

# akm_params['Q'] = 'cov(psi_t, psi_{t+1})'

f = fe.FEsolver(akm_params)

f.run_1()

# The next line is optional:
# f.construct_Q()
# Instead of running this function, we can manually generate Q:

# Modify adata to allow creation of Q matrix
# cov(alpha, psi)
# f.adata['Jq_row'] = f.adata['cs'] == 1
# f.adata['Jq_col'] = f.adata['f1i'] - 1
# f.adata['Wq_row'] = f.adata['cs'] == 1
# f.adata['Wq_col'] = f.adata['wid'] - 1

# To check results are equal:
# Jq.todense() - f.Jq.todense()
# Wq.todense() - f.Wq.todense()

# # cov(psi_t, psi_{t+1})
# f.adata['Jq_row'] = (f.adata['m'] == 1) & (f.adata['cs'] == 1)
# f.adata['Jq_col'] = f.adata['f1i'] - 1
# f.adata['Wq_row'] = (f.adata['m'] == 1) & (f.adata['cs'] == 0)
# f.adata['Wq_col'] = f.adata['f1i'] - 1

# To check results are equal:
# Jq.todense() - f.J1.todense()
# Wq.todense() - f.J2.todense()

f.construct_Q()

f.run_2()
