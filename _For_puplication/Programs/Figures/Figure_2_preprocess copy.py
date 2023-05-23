#%%
"""
Predicts on simulated data contained in sim_becnch folder and outputs to 
Output/Stdr_sim

The results is allready computed and saved in the folder,
but can be recomputed by the code below. 
full treatment time est. 1h 20 min.
"""
import sys
sys.path.append('../../../')
from Scripts.SEMORE_clustering import find_clust
#%%
path = '../../Div_data/sim_bench/{}'
save_path = '../../Output/Stdr_sim/{}'

settings_fibril = dict(
            save = save_path.format('Fib_sim'),
            plot = False,
            rough_min_points = 0, 
            final_min_points = 100,
            investigate_min_sample = 25,
            radius_ratio = 1,
            HDBSCAN_init = {
                            'min_cluster_size': 60,
                            'min_samples': 30,
                            'cluster_selection_method': 'eom',
                            'cluster_selection_epsilon': .0},)
settings_rand = dict(
        save = save_path.format('Rand_sim'),
        plot = False,
        investigate_min_sample = 25)

settings_iso = dict(
        save = save_path.format('Iso_sim'),
        plot = False,
        HDBSCAN_init = {
            'min_cluster_size': 100,
            'min_samples': 20,
            'cluster_selection_method': 'eom',
            'cluster_selection_epsilon': .05},
        investigate_min_sample = 25)


results = find_clust(path.format('Fib_sim'),**settings_fibril) #Run time ~20 min
results = find_clust(path.format('Rand_sim'),**settings_rand) #Run time ~35 min
results = find_clust(path.format('Iso_sim'),**settings_iso) #Run time ~26 min
# %%
