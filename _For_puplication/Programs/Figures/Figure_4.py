#%%
"""
Scripts for performing UAMP cluster on the collective features of the Insulin data set. 
However, UMAP random_state are dependent on system hardware, and same results are not guaranteed.

Look into the file "collective_feature/control_found_type.csv" for the correct cluster labels from figure 4.
"""
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import umap
#%%
feat_data_og= pd.read_csv('../../collective_features/Insulin.csv')
clust_nr = feat_data_og['Clust_nr']
feat_data = feat_data_og.drop(
    columns = ['Clust_nr','Ligand','FW'],
    inplace = False,
    errors = 'ignore')

morphology =['Variance', 'Circularity', 'Convexity', 'Circ_inertia']
morph_index = [feat_data.columns.get_loc(col) for col in morphology]
morph_index = range(len(feat_data.columns))
#%%
scaler = StandardScaler()
scaled_data = scaler.fit_transform(feat_data[morphology])
scaled_data = pd.DataFrame(scaled_data,columns=morphology)
reducer = umap.UMAP(
    min_dist=0.1,
    n_neighbors=10,
    random_state=42,
)
embedding = reducer.fit_transform(scaled_data)
model = DBSCAN(eps = .7, min_samples = 5)
model.fit(embedding)
#%%
fig,ax = plt.subplots(figsize=(10,10))
ax.scatter( embedding[:,0], embedding[:,1],c = model.labels_,cmap='Spectral')
# %%
