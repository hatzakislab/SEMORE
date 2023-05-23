#%%
import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.cluster import DBSCAN
from umap import UMAP

from Scripts.SEMORE_fingerprint import Morphology_fingerprint
"""
This scripts take a manually given csv file (stdr_sim_loose / stdr_sim_strict) and performs a umap as seen in figure 3.
"""
#%%
df_end = pd.read_csv('../../collective_features/stdr_sim_loose.csv')
lab_encoder = LabelEncoder().fit_transform(df_end.Ligand)

scaled_data = StandardScaler().fit_transform(df_end[df_end.columns[:-3]])
decom_umap = UMAP(
    min_dist = 0.1,
    n_neighbors =5,
    local_connectivity = 1,
    repulsion_strength = 1,
    n_components=3,
    random_state = 42,
    metric = 'euclidean',)
    
embedding = decom_umap.fit_transform(scaled_data)
all_cluster = DBSCAN(eps = .9)
all_cluster.fit(embedding)
#%%
cmap = clr.LinearSegmentedColormap.from_list('Nikos_staple', [
    (.0,'gray'),
    (.4,'lightgray'),
    (.6,'firebrick'),
    (1,'maroon')
    ], N=256)

figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot(111, projection='3d')
ax.scatter(
    xs = embedding[:,0]-5,
    ys = embedding[:,1]-3,
    zs = embedding[:,2],
    c = all_cluster.labels_,
    cmap = cmap,
    s = 100,
    edgecolors = 'black',
    alpha = .9,
)


ax.set_xlabel('UMAP 1',fontsize=20)
ax.set_ylabel('UMAP 2',fontsize=20)
ax.set_zlabel('UMAP 3',fontsize=20)

ax.view_init(20, 30)
#Set ratio of axis
ax.set_box_aspect((2,1,1))
#Set the zeroline black
ax.zaxis.line.set_color('black')
ax.xaxis.line.set_color('black')
ax.yaxis.line.set_color('black')
# Set the pane colors to aliceblue
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.set_facecolor('aliceblue')
ax.yaxis.pane.set_facecolor('aliceblue')
ax.zaxis.pane.set_facecolor('aliceblue')
#Set axis limits
ax.set_xlim(embedding[:,0].min()*0.9,embedding[:,0].max()*1)
ax.set_ylim(embedding[:,1].min()*0.9,embedding[:,1].max()*1.3)
ax.set_zlim(embedding[:,2].min()*0.9,embedding[:,2].max()*1.3)
#Plot black lines following the edges of the pane in the background
ax.plot(
    [embedding[:,0].min()*0.9,embedding[:,0].max()*1.05],
    [embedding[:,1].min()*0.9,embedding[:,1].min()*0.9],
    [embedding[:,2].min()*0.9,embedding[:,2].min()*0.9],
    color='black',linewidth=2)
ax.plot(
    [embedding[:,0].min()*0.9,embedding[:,0].min()*0.9],
    [embedding[:,1].min()*0.9,embedding[:,1].max()*1.4],
    [embedding[:,2].min()*0.9,embedding[:,2].min()*0.9],
    color='black',linewidth=2)
ax.plot(
    [embedding[:,0].min()*0.9,embedding[:,0].min()*.9],
    [embedding[:,1].min()*0.9,embedding[:,1].min()*.9],
    [embedding[:,2].min()*0.9,embedding[:,2].max()*1.3],
    color='black',linewidth=2)
#Plot a 2D projection unto the x-y plane
ax.scatter(
    xs = embedding[:,0]-5,
    ys = embedding[:,1]-3,
    zs = 0,
    color = 'black',
    alpha = .01,
    edgecolors = None,)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.rcParams.update({'font.size': 22})
#Plot a 2D projection unto the x-z plane

# %%
figure.savefig('../../Press/Fig/Fib_UMAP_3D_sim.png',dpi=300,bbox_inches='tight')
#%%

clust_nr = df_end.Clust_nr
clust_id = df_end[['Clust_nr','FW', 'Ligand']]
clust_id['UMAP_ID'] = all_cluster.labels_

for i in clust_id.UMAP_ID.unique():
    fig,ax = plt.subplots(10,4,figsize = (10,20),dpi = 150)
    fig.suptitle('UMAP ID: ' + str(i))
    fig.subplots_adjust(top=0.8)
    #ax.axis('off')
    ax = ax.flatten()
    for j in range(np.min([np.sum(clust_id.UMAP_ID == i),40])):
        temp_df = clust_id[clust_id.UMAP_ID == i].iloc[j]
        root_path = '../../Output/{}/Found_clusters/big_csv/'.format(temp_df.Ligand)
        data = pd.read_csv(root_path + temp_df.FW.split('\\')[-1] + '.csv')
        data = data[data.Final_label == temp_df.Clust_nr]
        ax[j].scatter(data['x'],data['y'],s = 20,c = data.frame,cmap = 'inferno')
        ax[j].set_title(temp_df.FW.split('\\')[-1] + ' ' + str(temp_df.Clust_nr))
        ax[j].axis('off')

    fig.tight_layout()
#%%
df_end['Pred_type'] = clust_id['UMAP_ID'].map({0:'Fib_sim', 1 :'Rand_sim', 2 : 'Iso_sim'})
df_end.to_csv('../../Output/collective_features/sim_found_type.csv',index = False)
# %%
clust_id['Ligand'] = df_end['Ligand']
clust_id[['Ligand','UMAP_ID']].value_counts()
for_heat = clust_id[clust_id.UMAP_ID != 1][['Ligand','UMAP_ID']]
for_heat['prediction'] = for_heat.UMAP_ID.map({0:'Fib_sim',1:'Noise', 2 :'Rand_sim', 3 : 'Iso_sim'})
for_heat = for_heat[['Ligand','prediction']]
conf_mat = confusion_matrix(
    for_heat['Ligand'],
    for_heat['prediction'],
    normalize='true',
    labels=['Fib_sim','Rand_sim','Iso_sim'])

f1 = f1_score(
    *for_heat.values.T,
    average=None,
    labels=['Fib_sim','Rand_sim','Iso_sim'])

f1_mean = f1_score(
    *for_heat.values.T,
    average='macro',
    labels=['Fib_sim','Rand_sim','Iso_sim'])
    
sns.heatmap(
    conf_mat,
    annot=True,
    xticklabels=['Fib_sim','Rand_sim','Iso_sim'],
    yticklabels=['Fib_sim','Rand_sim','Iso_sim'],
    fmt='.2%',
    cmap='Reds',
    cbar=False,
    annot_kws={'size': 23.5},
    linewidths=1,
    linecolor='black',
    square=True,)
# %%