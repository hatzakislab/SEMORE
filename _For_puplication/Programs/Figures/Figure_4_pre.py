#%%
import sys
sys.path.append('../../../')
from Scripts.SEMORE_clustering import find_clust
from Scripts.SEMORE_fingerprint import Morphology_fingerprint
from glob import glob
import pandas as pd
from tqdm import tqdm
#%%
save_path ='../../Output/Insulin_data'
find_clust('../../Insulin_data',save = save_path,plot = False)
# %%
sim = 'Insulin'
path = '../../Output/Insulin_data/big_csv/*.csv'
files = glob(path)
df_end = pd.DataFrame()
df_skipped = pd.DataFrame()
for file in tqdm(files):
    prediction = pd.read_csv(file)
    prediction.rename(columns={'x [nm]': 'x','y [nm]': 'y'},inplace=True)
    for label,subset in prediction.groupby('Final_label'):
        if label == -1:
            continue
        data = subset[['frame','x','y']]
        cluster_feat = Morphology_fingerprint(data)
        fetures = cluster_feat.all_run()
        clust_id = {'Clust_nr': label, 'Ligand': sim, 'FW': file.split('/')[-1].split('.')[0]}
        fetures.update(clust_id)
        
        skipped = {'Clust_nr': label, 'Ligand': sim, 'FW': file.split('/')[-1].split('.')[0]}
        df_skipped = pd.concat([df_skipped,pd.DataFrame(skipped,index=[0])],ignore_index=True)

        df = pd.DataFrame(fetures)

        df_end = pd.concat([df_end,df],ignore_index = True)
df_end.to_csv('../../collective_features/Insulin.csv',index=False)
# %%
