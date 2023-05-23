"""
This scripts is used to generate the viewed plots in figure 1 of the article.
For the zoom in created in Figure 1.C, the cluster label 79 should be investigated, 
with the code marked as string applied.

For the indevidual squares in Figure 1.e.
SEMORE fingerprint class meethod "plot_feature" 
which is final_label == 3 when only investigating cluster label 79.
"""
import sys
sys.path.append('../../../')
from Scripts.Usefull_func import get_contur
from Scripts.SEMORE_fingerprint import Morphology_fingerprint
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
from scipy import spatial
from sklearn import cluster,preprocessing,neighbors
import warnings
import cv2 as cv
from tqdm import tqdm


font = {'weight' : 'bold',
    'size' : 20}
plt.rc('font', **font)

style = {'xtick.labelsize': 18,'ytick.labelsize':18,'font.size':18,'axes.labelweight': 'bold',
        'figure.dpi': 200}
mpl.RcParams(style)

warnings.filterwarnings('ignore',category = UserWarning)
data = pd.read_csv('../../Insulin_data/0206_3.csv')
data.rename(columns = {'x [nm]':'x','y [nm]': 'y'},inplace = True)
X = data
X = X.values[:,:3]
scaler = preprocessing.StandardScaler()
X_scale = scaler.fit_transform(X)
X_fist_fit = np.roll(X_scale,shift = 2,axis = 1)
model = hdbscan.HDBSCAN(min_cluster_size = 100,min_samples = 20,cluster_selection_method = 'eom',cluster_selection_epsilon = .03)
model.fit(X_fist_fit[:,:2])
labels, counts = np.unique(model.labels_[model.labels_ != -1],return_counts = True)
labels = list(labels)
labels.insert(0,labels.pop(np.argmax(counts)))
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)
ax.scatter(*X[model.labels_ != -1][:,1:3].T,s=0.5,c = model.labels_[model.labels_ != -1],cmap = 'tab20')
ax.scatter(*X[model.labels_ == -1][:,1:3].T,s=0.5,color = 'k')
ax.set_ylabel('y [nm]',{'fontweight': 'bold'})
ax.set_xlabel('x [nm]',{'fontweight': 'bold'})
plt.show()

final_labels = np.zeros(model.labels_.shape[0]) - 1
for lab in [79]:#tqdm(labels):
    X_closer = X[model.labels_ == lab]
    X_closer = np.roll(X_closer,shift = 2,axis = 1)

    X_closer_scale = preprocessing.MinMaxScaler().fit_transform(X_closer)
    current_labels = np.ones(len(X_closer_scale)) * -1

    distances = spatial.distance.cdist(np.array([[0,0,0]]),X_closer_scale)
    distance_low = np.percentile(distances,25)
    distance_high = np.percentile(distances,75)
    distances_insepect = distances[(distances>= distance_low) * (distances<= distance_high)]
    radius = np.sqrt(distances_insepect.std()/np.sqrt(len(distances_insepect)))*1.3
    

    model_nd = cluster.DBSCAN(eps = radius, 
                              min_samples = 50)


    model_clust = neighbors.RadiusNeighborsClassifier(radius = radius,
                                                      weights = 'distance',
                                                      outlier_label = -1)
    c = 0
    for i in np.unique(X_closer_scale[:,2]):
        frame_mask = X_closer_scale[:,2] <= i
        label_mask = current_labels == -1
        the_mask = frame_mask * label_mask
        X_eval = X_closer_scale[:,:2][the_mask]
        if the_mask.sum() == 0:
            continue
        NN_labels = np.ones(X_eval.shape[0]) * -1
        try:
            model_clust.fit(X_closer_scale[~label_mask],current_labels[~label_mask])
            NN_labels = model_clust.predict(X_closer_scale[the_mask])
        except:
            None
        try:
            model_nd.fit(X_eval[NN_labels == -1])
            model_nd.labels_[model_nd.labels_ != -1] += np.unique(current_labels[current_labels != -1]).shape[0]
            NN_labels[NN_labels == -1] = model_nd.labels_
        except:
            None
        current_labels[the_mask] = NN_labels
    """
        if i% (X_closer_scale[:,2].max()/5) == 0:
            fig = plt.figure(figsize = (12,12))
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.scatter(*X_closer_scale[frame_mask][:,:2].T, c = current_labels[frame_mask],s = 1,cmap = 'tab10',vmax = 22,vmin = -1)
            #fig.savefig('../../Press/Fig/Time_eval/frame_{}.png'.format(c))
            c+=1
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.scatter(*X_closer_scale[frame_mask][:,:2].T, c = current_labels[frame_mask],s = 1,cmap = 'tab10',vmax = 22,vmin = -1)
    #fig.savefig('../../Press/Fig/Time_eval/frame_{}.png'.format(c))
    """
    if np.all(current_labels == -1):
        current_labels += 1 
    
    current_labels[current_labels!=-1] += np.max([final_labels.max(),0])+1
    final_labels[model.labels_ == lab] = current_labels

l_, c_ = np.unique(final_labels,return_counts = True) 
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111)
for label in tqdm(l_):
    if label == -1:
        continue
    to_be_plot = X[:,1:3][final_labels == label]

    if len(to_be_plot)< 300:
        continue
    if label in l_[:22]:
        ax.scatter(*to_be_plot.T,s=1,c = final_labels[final_labels ==label],cmap = 'tab10',vmax = 23,vmin = 0)
    else:
        ax.scatter(*to_be_plot.T,s=1,cmap = 'tab10')

ax.set_ylabel('y [nm]',{'fontweight': 'bold'})
ax.set_xlabel('x [nm]',{'fontweight': 'bold'})
plt.show()

test = X[:,1:3][final_labels == 2]
morph_class = Morphology_fingerprint(data[final_labels == 2][['x','y','frame']])
morph_class.all_run()

fig,ax = plt.subplots(2,2,figsize = (20,20))
ax = ax.flatten()
for ax_ in ax:
    ax_.set(**{'xticklabels':[],'yticklabels':[]})
morph_class.plot_features('Circularity',ax[0],circ = True)
morph_class.plot_features('Graph network',ax[1],graph = True)
morph_class.plot_features('Symmetry',ax[2],sym = True)
morph_class.plot_features('Area',ax[3],triangle = True)
fig.tight_layout()
plt.show()
