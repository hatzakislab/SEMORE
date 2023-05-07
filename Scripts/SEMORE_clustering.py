#%%
import os
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy import spatial
from sklearn import cluster,preprocessing,neighbors
import warnings
from tqdm import tqdm
from .Usefull_func import *
warnings.filterwarnings('ignore',category = UserWarning)

def find_clust(
    path,
    rough_min_points:int = 100,
    final_min_points:int = 200,
    radius_ratio:float = 1.96,
    investigate_min_sample:int = 50,
    DBSCAN_init:dict = None,
    HDBSCAN_init:dict = None,
    init_scale_type:str = 'StandardScaler',
    dens_mode:str = 'strict',
    plot:bool = True,
    save:bool = False
    ) -> pd.DataFrame:
    """
    This function takes a path to a csv file or a folder containing csv files and performs a clustering on the data as described in the paper. The results are saved in a csv file and a plot is made.

    Parameters
    ----------
    path : str
        Path to a csv file or a folder containing csv files.
    rough_min_points : int, optional
        Minimum number of points for the rough clustering, by default 100
    final_min_points : int, optional
        Minimum number of points for the final clustering, by default 200
    radius_ratio : float, optional
        Radius ratio between the final clustering and the rough clustering, by default 1.96
    investigate_min_sample : int, optional
        Minimum number of points that a cluster must have to be investigated, by default 50
    DBSCAN_init : dict, optional
        Dictionary containing the overwriting-parameters for the DBSCAN algorithm, by default None
    HDBSCAN_init : dict, optional
        Dictionary containing the overwriting-parameters for the HDBSCAN algorithm, by default None
    init_scale_type : str, optional
        Type of scaler used to scale the data, by default 'StandardScaler'
    dens_mode : str, optional
        Determin whether the density_filter is strict (both density and final_min_ponts) or loose (either of them), by default 'strict'
    plot : bool, optional
        Determines whether the results are plotted, by default True
    save : bool or Str, optional
        Determines whether the results are saved. If Str is passed, this parameter is used to create folders for output at given path. If "auto" folders are created correspondingly to data-path, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the clustering.
    """

    data = None
    cluster_count = 0
    if not path.split('.')[-1] == 'csv':
        file = glob(path + '/*.csv')
        folder_name = path.split('/')[-1]
        follow = path.split('/')[:-2]
    else:
        file = [path]
        folder_name = path.split('/')[-2]
        follow = path.split('/')[:-3]

    if save is not False:
        if save == 'Auto':
            save_path = '/'.join(follow) + '/Output/'+ folder_name 
        else:
            save_path = save

        if not os.path.isdir(save_path):     
            os.makedirs(save_path)
            os.mkdir(save_path + '/csv_files')
            os.mkdir(save_path + '/plots')
            os.mkdir(save_path + '/big_csv')
    o = 0
    for k in file:
        print('At file nr {}/{}'.format(o+1,len(file)))
        o+=1
        k = k.replace('\\','/')

        if DBSCAN_init == None:
            DBSCAN_init = {'eps':.09,'n_jobs': -1,'min_samples': 50}
        if HDBSCAN_init == None:
            HDBSCAN_init = {'min_cluster_size': 100,'min_samples': 20,'cluster_selection_method': 'eom','cluster_selection_epsilon': .03}
        if init_scale_type == 'StandardScaler':
            scaler = preprocessing.StandardScaler()
        elif init_scale_type == 'MinMaxScaler':
            scaler = preprocessing.MinMaxScaler()
        else:
            raise ValueError('init_scale_type must be either StandardScaler or MinMaxScaler got {}'.format(init_scale_type))

        print('Running on file: {}'.format(k.split('/')[-1]))
        data = pd.read_csv(k)
        try:
            x = data['x [nm]']
            y = data['y [nm]']
        except:
            x = data['x']
            y = data['y']

        frames = data['frame']
        X = np.array([x,y,frames]).T
        X_scale = scaler.fit_transform(X)

        min_dens = X_scale.shape[0]/np.product(np.diff([X_scale[:,:2].min(axis = 0) ,
                                                          X_scale[:,:2].max(axis = 0)],
                                                          axis = 0))
        min_dens = np.min((min_dens,3000))
        model = hdbscan.HDBSCAN(**HDBSCAN_init)
        if min_dens < 1500: model = cluster.DBSCAN(**DBSCAN_init)

        model.fit(X_scale[:,:2])

        if np.unique(model.labels_).shape[0] == 1:
            mask = find_atleast_one_init_clust(X)
            model.labels_[mask] = 0

        noise_points = X_scale[model.labels_ == -1][:,:2]
        min_dens,dense_multi = estimate_noise_density(noise_points)

        l_init,c_init = np.unique(model.labels_[model.labels_ != -1],return_counts = True)
        data['Rough_label'] = model.labels_

        print('Min_dense',min_dens,'Ratio',dense_multi)

        if plot:
            figure = plt.figure(figsize = (12,8))
            ax = figure.add_subplot(111)
            ax.scatter(*X[model.labels_ != -1][:,:2].T,s=0.1,c = model.labels_[model.labels_ != -1],cmap = 'Set3')
            ax.scatter(*X[model.labels_ == -1][:,:2].T,s=0.1,color = 'k',alpha = .5)
            ax.set_xlabel("x [nm]")
            ax.set_ylabel("y [nm]")
            if save:
                figure.savefig(save_path + '/plots' + '/{}_rough_plot.pdf'.format(k.split('/')[-1].split('.')[0]))

        final_label = np.zeros(model.labels_.shape[0])-1
        for lab in tqdm(l_init[c_init > rough_min_points]):
            X_closer = X[model.labels_ == lab]

            X_closer_scale = preprocessing.MinMaxScaler().fit_transform(X_closer)
            current_labels = np.ones(len(X_closer_scale)) * -1

            distances = spatial.distance.pdist(X_closer_scale[:,:3])
            distance_low = np.percentile(distances,25)
            distance_high = np.percentile(distances,75)
            distances_insepect = distances[(distances>= distance_low) * (distances<= distance_high)]
            radius = np.sqrt(np.std(distances_insepect)/np.sqrt(X_closer_scale.shape[0])*radius_ratio)
            model_nd = cluster.DBSCAN(eps = radius, 
                                    min_samples = investigate_min_sample)

            model_clust = neighbors.RadiusNeighborsClassifier(radius = radius,
                                                            weights = 'distance',
                                                            outlier_label = -1)

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

            if np.all(current_labels == -1):
                current_labels += 1 
            current_labels[current_labels!=-1] += np.max([final_label.max(),0])+1
            final_label[model.labels_ == lab] = current_labels

        fin_l,fin_c = np.unique(final_label,return_counts = True)
        fig = plt.figure(figsize = (20,8))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        ax.set_xlabel('x[nm]')
        ax.set_ylabel('y[nm]')

        ax1.set_xlabel('x[nm]')
        ax1.set_ylabel('y[nm]')

        for count,label in zip(fin_c,fin_l):
            saveabel = True
            mask = final_label == label
            if label == -1:
                continue
            hull = spatial.ConvexHull(X_scale[:,:2][mask])
            dens = count/hull.volume
            multi = 1
            if type(model) == hdbscan.HDBSCAN: multi = dense_multi
            style_dict = {'c': 'k',
                          'ls' : '-',
                          'lw': 2}
            con_col = 'k'
            if dens_mode == 'loose':
                checks = dens < multi * min_dens and count < final_min_points
            elif dens_mode == 'strict':
                checks = dens < multi * min_dens or count < final_min_points
            elif dens_mode == None:
                checks = False

            if checks:
                final_label[mask] = -1
                style_dict = {'c': 'red',
                             'ls': '--',
                             'lw': 1,
                             'alpha': 0.5}
                saveabel = False


            x,y =  X[mask][:,:2].T
            frame = frames[mask]

            cluster_dict = {'frame': frame,'x':x,'y':y}
            ax.scatter(x,y,s=0.5,cmap = 'Set20')
            ax.plot(x[hull.vertices], y[hull.vertices], **style_dict)
            ax.plot(x[np.roll(hull.vertices,1)], y[np.roll(hull.vertices,1)], **style_dict)

            cluster_df = pd.DataFrame(cluster_dict)
            if save and saveabel: cluster_df.to_csv(save_path + '/csv_files' + '/{}.csv'.format('Cluster_{}_'.format(int(label + cluster_count))+ k.split('/')[-1][:-4]), index = False)
        cluster_count+= len(np.unique(final_label)) - 1
        data['Final_label'] = final_label
        time = ax1.scatter(*X[:,:2].T,s=0.2,c = frames,cmap = 'inferno_r')
        ax.scatter(*X[:,:2].T,s=0.2,c = 'k',
        alpha = 0.4,zorder = 0)
        plt.colorbar(time,ax = ax1,label = 'Frame')
        if save:
            fig.savefig(save_path + '/plots' + '/{}_cluster_plot.pdf'.format(k.split('/')[-1].split('.')[0]))
            fig.savefig(save_path + '/plots' + '/{}_cluster_plot.png'.format(k.split('/')[-1].split('.')[0]))
            data.to_csv(save_path + '/big_csv' + '/{}.csv'.format(k.split('/')[-1].split('.')[0] + '_CLF'),index = False)
        if not plot:
                plt.close()
    return data
# %%
