'''
Tests Attrition class.
'''
import pytest
import numpy as np
import bipartitepandas as bpd
import pytwoway as tw

#####################
##### Attrition #####
#####################

def test_attrition_1():
    # Test attrition_increasing() and attrition_decreasing(), by checking that the fraction of movers remaining is approximately equal to what `subsets` specifies.
    rng = np.random.default_rng(6481)

    # First test increasing, then decreasing
    subsets_lst = [np.linspace(0.1, 1, 10), np.linspace(1, 0.1, 10)]
    attrition_fn = [tw.attrition_utils.AttritionIncreasing, tw.attrition_utils.AttritionDecreasing]

    # Set some parameters
    sim_network = bpd.SimBipartite()
    clean_params = bpd.clean_params({'verbose': False})
    convert_params = {'is_sorted': True, 'copy': False}
    tol = [1e-4, 1e-3]

    for i in range(2):
        # Non-collapsed
        bdf = bpd.BipartiteDataFrame(sim_network.simulate(rng), track_id_changes=True).clean(clean_params).to_eventstudy(**convert_params)

        orig_n_movers = bdf.loc[bdf['m'] > 0, :].n_workers()

        mover_subsets = attrition_fn[i](subsets_lst[i])._gen_subsets(bdf, clean_params=clean_params, rng=np.random.default_rng(1234))
        mover_subsets = np.array([orig_n_movers - len(mover_subset[1]) if mover_subset[1] is not None else orig_n_movers for mover_subset in mover_subsets])

        attrition_vs_input = abs((mover_subsets / orig_n_movers) - subsets_lst[i])

        assert np.max(attrition_vs_input) < tol[i]

        # Collapsed
        bdf = bpd.BipartiteDataFrame(sim_network.simulate(rng), track_id_changes=True).clean(clean_params).collapse(**convert_params).to_eventstudy(**convert_params)

        orig_n_movers = bdf.loc[bdf['m'] > 0, :].n_workers()

        mover_subsets = attrition_fn[i](subsets_lst[i])._gen_subsets(bdf, clean_params=clean_params, rng=np.random.default_rng(1234))
        mover_subsets = np.array([orig_n_movers - len(mover_subset[1]) if mover_subset[1] is not None else orig_n_movers for mover_subset in mover_subsets])

        attrition_vs_input = abs((mover_subsets / orig_n_movers) - subsets_lst[i])

        assert np.max(attrition_vs_input) < tol[i]
