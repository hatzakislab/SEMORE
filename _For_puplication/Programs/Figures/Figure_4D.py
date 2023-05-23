#%%
"""
This script is used to segmentate and treat the data for Figure 4D. 
Again it is important to remember that UMAP is depedent on the random state, 
and therefor no garantee that the results will be the same as in the manuscript.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys 
sys.path.append('../../../')
import seaborn as sns
from sklearn import cluster, preprocessing
from Scripts.SEMORE_fingerprint import Morphology_fingerprint
from tqdm import tqdm

plt.rcParams.update({'font.size': 20})
# %%
nipy = plt.get_cmap('nipy_spectral',3)
data = '../../Div_data/SMLM/Nanopore_real.csv'
df = pd.read_csv(data)
df[['x', 'y']] *= (132,142)
#%%
df = df[np.logical_and(df.x>5000, df.x<12500)]
df = df[np.logical_and(df.y>5000, df.y<12500)]
model = cluster.DBSCAN(eps= 50, min_samples=10)

model.fit(df[['x', 'y']])

df_signal = df[model.labels_!=-1]
labes_signal = model.labels_[model.labels_!=-1]


l_,c_ = np.unique(labes_signal, return_counts=True)

filter = c_>0
l_ = l_[filter]
filtered_labels = np.isin(labes_signal, l_)
figure = plt.figure(figsize=(10,10),dpi = 200)
ax = figure.add_subplot(111)
ax.scatter(
    df['x'],
    df['y'],
    c = 'k',
    alpha=1,
    s= 3
)
ax.scatter(
    df_signal['x'], 
    df_signal['y'], 
    c = labes_signal,
    s= 3,
    cmap = 'tab20') 
ax.set_aspect('equal')
ax.set_xlim(5000,12500)
ax.set_ylim(5000,12500)
ax.set_xticks([])
ax.set_yticks([])
# %%
spat_dict = {'n_gauss': 4}
df_nanopore = pd.DataFrame()
cores = np.empty(shape=(len(l_),2))
for i in tqdm(l_):
    subset = df_signal[labes_signal==i]
    clust = Morphology_fingerprint(subset,start = 'mid',cut_param = .95)
    try:
        features = clust.all_run(spatial_param = spat_dict)
        cores[i] = clust.start_core
    except:
        print('No features for cluster ',i,'.')
        features = pd.DataFrame()
    features['label'] = i
    subdf = pd.DataFrame(features)
    df_nanopore = pd.concat([df_nanopore, subdf], axis = 0)
# %%
scaler = preprocessing.StandardScaler()
skip_columns = df_nanopore.columns[[np.isin(['mu','sig','w','State'], x.split('_')).max() for x in df_nanopore.columns]]
use_columns = [x for x in df_nanopore.columns if x not in skip_columns]

df_scaled = scaler.fit_transform(df_nanopore[use_columns[:-1]])

from umap import UMAP
reducer = UMAP(
    n_neighbors=10,
    min_dist=0.2,
    random_state = 42,
    n_components=2,
    metric = 'euclidean',
)
embedding = reducer.fit_transform(df_scaled)
#sns.set_style({'axes.grid' : True, 'axes.facecolor':'aliceblue'})
figure = plt.figure(figsize=(10,10))
ax = figure.add_subplot(111)
umap_model = cluster.DBSCAN(eps = .6,min_samples = 5)
umap_model.fit(embedding)
ax.scatter(
    embedding[:,0], 
    embedding[:,1], 
    c = umap_model.labels_,
    s= 20, alpha=1,
    cmap = 'nipy_spectral',
    edgecolors = 'k',
    linewidths = 0.5
    )
ax.set_aspect('auto')
ax.set_xticklabels([])
ax.set_yticklabels([])
# %%
for umap_l in np.unique(umap_model.labels_):
    fig,ax = plt.subplots(5,5,figsize=(10,10))
    ax = ax.flatten()
    fig.suptitle(f'Cluster {umap_l}')
    subset = df_nanopore[umap_model.labels_==umap_l]
    for a,l in enumerate(subset.label.unique()):
        subsubset = df_signal[labes_signal==l]
        if a < len(ax):
            sns.scatterplot(
                data = subsubset, 
                x = 'x', 
                y = 'y',
                ax = ax[a])
            ax[a].set_aspect('equal')
            ax[a].set_xticklabels([])
            ax[a].set_yticklabels([])
            ax[a].set_ylabel('')
            ax[a].set_xlabel('')
        else:
            pass    
# %%
#connect labels to cordinates
df_nanopore['Umap_label'] = umap_model.labels_
grp_labels = df_nanopore.groupby('Umap_label').label
grp_dict = grp_labels.apply(lambda x: x.unique()).to_dict()
if -1 not in grp_dict.keys():
    grp_dict[-1] = np.array([-1])

def give_umap(id):
    try:
        index = np.argwhere([id in x[-1] for x in grp_dict.items()]).ravel()[0]
        key = list(grp_dict.keys())[index]
    except:
        key = 0
    return key

df_signal['cluster_label'] = labes_signal
df_signal['Umap_label'] = df_signal.cluster_label.map(give_umap)
# %%
figure = plt.figure(figsize=(10,10))
ax2 = figure.add_subplot(111)

ax2.scatter(
    df_signal.x, 
    df_signal.y, 
    c = df_signal.Umap_label, 
    s = 6,
    cmap = 'nipy_spectral',
    edgecolors = 'k',
    linewidths = 0.1)

ax2.set_aspect('auto')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')
ax2.set_xlim(5000,12500)
ax2.set_ylim(5000,12500)
# %%
focus_ = np.sqrt(df_nanopore[df_nanopore.Umap_label == 1]/np.pi)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
x = np.linspace(10, 100, 1000)
one_gauss = stats.norm.fit(focus_.Area)
ax.plot(x, stats.norm.pdf(x, *one_gauss), 'k', lw = 3)

_ = ax.hist(focus_.Area, bins = 20, density = True, alpha = 0.5,color = nipy(1))
_ = ax.hist(focus_.Area, bins = 20, density = True, histtype = 'step', color = 'k')


ax.set_xlabel('Radius [nm]')
ax.set_ylabel('Probability density')
ax.set_yticks([])

textstr = '\n'.join((
    r'N=%.0f' %(focus_.shape[0]),
    r'$\mu=%.2f nm$' % (one_gauss[0]),
    r'$\sigma=%.2f nm$' % (one_gauss[1])
    ))
ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top')




# %%
