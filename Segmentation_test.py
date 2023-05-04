#%%
from Scripts.SEMORE_clustering import find_clust

# %%
path = 'Test_data/simulation_test.csv'
output = 'Output/simulation_test'
results = find_clust(path,save = output,final_min_points= 100)
# %%
