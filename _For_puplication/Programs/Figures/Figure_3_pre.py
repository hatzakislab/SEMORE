import pandas as pd
import numpy as np
from glob import glob
import sys
sys.path.append('../../')
from Scripts.SEMORE_fingerprint import Morphology_fingerprint
from tqdm import tqdm
"""
This scripts extract hte fingerprints used to create the umap seen in figure 3. Run time 30min
"""
simulations = ['Fib_sim','Iso_sim','Rand_sim']
root_path = '../../Output/Loose_filter/'
#root_path = '../../Output/Stdr_sim/'
df_end = pd.DataFrame()
df_skipped = pd.DataFrame()
for sim in simulations:
    path = root_path + sim + '/big_csv/*.csv'
    files = glob(path)
    for file in tqdm(files):
        prediction = pd.read_csv(file)
        for label in prediction.Final_label.unique():
            if label == -1:
                continue
            data = prediction[prediction.Final_label == label ][['frame','x','y']]
            cluster_feat = Morphology_fingerprint(data)
            fetures = cluster_feat.all_run()
            clust_id = {'Clust_nr': label, 'Ligand': sim, 'FW': file.split('/')[-1].split('.')[0]}
            fetures.update(clust_id)
            
            skipped = {'Clust_nr': label, 'Ligand': sim, 'FW': file.split('/')[-1].split('.')[0]}
            df_skipped = pd.concat([df_skipped,pd.DataFrame(skipped,index=[0])],ignore_index=True)

            df = pd.DataFrame(fetures)

            df_end = pd.concat([df_end,df],ignore_index = True)
df_end.to_csv('../../collective_features/stdr_sim_loose.csv',index=False)