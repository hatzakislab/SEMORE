#%%
from Scripts.SEMORE_fingerprint import Morphology_fingerprint
import pandas as pd
# %%
path = 'Output/simulation_test/big_csv/simulation_test_CLF.csv'
output = 'Output/simulation_test/'

data = pd.read_csv(path)
#%%
#Extract fingerprints from each group
grp = data[['x','y','frame','Final_label']].query('Final_label != -1').groupby('Final_label')
final_df = pd.DataFrame()
for sub_set in grp:
    aggregate_class = Morphology_fingerprint(sub_set[1])
    features = aggregate_class.all_run()
    feat_df = pd.DataFrame(features,index = [0])
    feat_df['label'] = sub_set[0]
    final_df = pd.concat([final_df,feat_df],axis = 0)
print(final_df)
# %%
aggregate_class.plot_features(
    triangle = True,
    circ = True,
    sym = True,
    graph = True,)
# %%
