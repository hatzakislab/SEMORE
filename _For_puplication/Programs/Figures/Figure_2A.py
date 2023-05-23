#%%
"""
This scripts is used to visualize the representative results seen in Figure 2A of the article.
The 3D version of the wanted aspect is procudes in the end script. Manual labor is required to change experiment and 
3D visulaization. 
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import seaborn as sns
from glob import glob
import pandas as pd
import sys
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
sys.path.append('../../../')
from Scripts.Usefull_func import fix_labels, find_acc

#Import predicted data
prediction = pd.read_csv('../../Output/Stdr_sim/Iso_sim/big_csv/Iso_10_04_0_CLF.csv')
prediction = prediction.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'],errors='ignore')
#%%
#Fixing the labels
prediction['Final_label'] = fix_labels(prediction.Final_label,prediction.label)
prediction['Final_label'] = prediction.Final_label.astype(int)
marker_true = prediction.Final_label.values == prediction.label
# %%
figure, ax = plt.subplots(1,3,figsize=(30,10))
# remove axis ticks and labels
for axis in figure.get_axes():
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel('')
    axis.set_ylabel('')
#Set titles
ax[0].set_title('Raw data',fontsize=20)
ax[1].set_title('True labels',fontsize=20)
ax[2].set_title('Predicted labels',fontsize=20)
#First ax contains raw data, second true labels and last final_labels
sns.scatterplot(
    data=prediction,
    x='x',
    y='y',
    s = 5,
    ax = ax[0],
    edgecolor = None,
    color = 'k'
)
sns.scatterplot(
    data=prediction[prediction.label != -1],
    x='x',
    y='y',
    s = 10,
    ax = ax[1],
    hue = 'label',
    palette = 'tab20',
    edgecolor = None,
    legend = False,
    vmax = 10,
    vmin = 0
)

sns.scatterplot(
    data=prediction.query('Final_label != -1 & Final_label == label'),
    x='x',
    y='y',
    s = 10,
    ax = ax[2],
    hue = 'Final_label',
    palette = 'tab20',
    edgecolor = None,
    legend = False
)
sns.scatterplot(
    data = prediction[(prediction.Final_label == -1) * (~marker_true)],
    x='x',
    y='y',
    s = 10,
    ax = ax[2],
    color = 'brown',
    edgecolor = 'black',
    linewidth = 0,
    label = 'Missed points',
    zorder = 0
)
sns.scatterplot(
    data = prediction[(prediction.Final_label != -1) * (~marker_true)],
    x='x',
    y='y',
    s = 10,
    ax = ax[2],
    color = 'k',
    label = 'Wrongly assigned points',
    edgecolor = 'white',
    zorder = 0
)
# Setting axis limits to be equal
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
ax[2].set_xlim(ax[0].get_xlim())
ax[2].set_ylim(ax[0].get_ylim())

ax[2].legend(markerscale = 10, fontsize = 20, loc = 'upper right')
#%%
figure = plt.figure(figsize= (40,20))
ax = figure.add_subplot(111, projection='3d')
ax.view_init(20, 60)
to_plot = prediction
colors = 'k'
from matplotlib.patches import Rectangle,PathPatch
ax.scatter(to_plot.x+4500,to_plot.frame,to_plot.y+6200,s=15,c=colors,cmap='tab20')
ax.scatter(to_plot.x+4500,500,to_plot.y+6200,s=15,c=colors,cmap='tab20')
ax.plot([60000,60000],[-20,500],[60000,60000],c='k',linewidth=5,zorder = 10,ls = '--')
ax.plot([-60000,-60000],[-20,500],[60000,60000],c='k',linewidth=5,zorder = 10,ls = '--')
ax.plot([60000,-60000],[-20,-20],[60000,60000],c='k',linewidth=5,zorder = 10,ls = '--')
ax.plot([-60000,-60000],[-20,-20],[60000,-60000],c='k',linewidth=5,zorder = 0,ls = '--')
ax.plot([60000,-60000],[-20,-20],[-60000,-60000],c='k',linewidth=5,zorder = 0,ls = '--')
ax.plot([60000,60000],[-20,-20],[-60000,60000],c='k',linewidth=5,zorder = 0,ls = '--')
ax.plot([60000,60000],[-20,500],[-60000,-60000],c='k',linewidth=5,zorder = 0,ls = '--')
ax.plot([-60000,-60000],[-20,500],[-60000,-60000],c='k',linewidth=5,zorder = 0,ls = '--')

square = Rectangle((-60000, -60000), 120000, 120000, fc="w", fill=True, edgecolor='black', linewidth=5)
ax.add_patch(square)
art3d.pathpatch_2d_to_3d(square, z=500, zdir="y")


ax.set_ylim(0,500)
ax.set_xlim(-58000,58000)


ax.invert_xaxis()
ax.set_zlim(-58000,58000)
ax.set_ylabel('Time',fontsize = 64,rotation = -40)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.xaxis.pane.fill = False
ax.zaxis.pane.fill = False


ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('w')
#ax.grid(False, axis = 'both')
# %%
