import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from scipy import ndimage,sparse

def get_contur (points,sigma = 5,percentile = 60,bins = 50, weights = None):
    if weights is None:
        weights = np.ones(points.shape[0])
    x_bins = np.linspace(points[:,0].min(),points[:,0].max(),bins + 2*sigma + 2)
    y_bins = np.linspace(points[:,1].min(),points[:,1].max(),bins + 2*sigma + 2)
    h,x_edge,y_edge= np.histogram2d(*points.T,bins = bins,density = True,weights = weights)
    h = np.pad(h,pad_width = (sigma,sigma), mode = 'constant', constant_values = 0)
    blured = ndimage.gaussian_filter(h,sigma = sigma)
    blured[blured < np.percentile(blured,percentile)] = 0  
    blured[blured >= np.percentile(blured,percentile)] = 1 
    blured = np.pad(blured,pad_width = (1,1), mode = 'constant', constant_values = 0)
    pcs = plt.contour(x_bins,y_bins,blured.T,levels = [.5])
    plt.close()
    cords = np.empty((1,2))
    for item in pcs.collections:
        best_v = []
        for i in item.get_paths():
            if len(i.vertices) > len(best_v):
                v = i.vertices
                best_v = v
        cords = np.append(cords,best_v,axis = 0)
    cords = cords[1:]
    return cords,pcs

def get_graph_cords(X,graph):
    X = np.asarray(X)

    G = sparse.coo_matrix(graph)
    A = X[G.row].T
    B = X[G.col].T

    x = np.vstack([A[0], B[0]])
    y = np.vstack([A[1], B[1]])
    return x,y

def get_graph_road(start,end,predecessors):
    nex_ind = [end]
    while nex_ind[-1] != -9999:
        next_ = predecessors[(start,nex_ind[-1])]
        nex_ind.append(next_)
    index_plot = list(zip(np.roll(nex_ind,0),np.roll(nex_ind,-1)))[:-2]
    index_plot = np.reshape(index_plot,(-1,2))
    return index_plot

def PArea (x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def fix_labels (prediction,true_label):
    test = prediction*1000 + 100
    l_,c_ = np.unique(true_label,return_counts=True)
    l_ = np.asarray(sorted(l_,key=lambda x: c_[l_ == x],reverse=True))[::-1]
    for label in l_:
        lab,counts = np.unique(prediction[true_label == label],return_counts=True)
        highest = lab[np.argmax(counts)]
        if highest == -1:
            continue
        test[prediction == highest] = label
    try:
        test[test <= -1] = -1
        #test[test >= 200] = 999
    except:
        None
    return test

def metric(grp,prediction_label = 'Final_label'):
    try:
        return_dict = dict(
            TPred = ((grp[prediction_label] == grp.label) * grp.weight).sum(),
            FPred = ((grp[prediction_label] != grp.label) * grp.weight).sum()
        )
    except:
        return_dict = dict(
            TPred = None,
            FPred = None
        )
    return pd.Series(return_dict)

def find_acc(data,true_col:str,pred_col:str,noise_mask = True):
        temp_all_labels = np.unique(np.append(data[true_col],data[pred_col]))
        agg_acc = pd.DataFrame()
        for lab_ in temp_all_labels:
            temp_query = data.query(f'{pred_col} == @lab_')
            temp_true_query = data.query(f'{true_col} == @lab_')
            label_N = temp_true_query.shape[0]
            pred_true = (temp_query[true_col] == temp_query[pred_col]).sum()
            if pred_true == 0 and temp_query.shape[0] > 0:
                acc = 0
            elif pred_true == 0 and label_N == 0:
                acc = 1
            else:
                acc = pred_true/np.max([label_N,temp_query.shape[0]])
            if noise_mask and lab_ == -1:
                report_label = 'Noise'
            elif noise_mask and lab_ > 200:
                report_label = 'Hallu'
            elif noise_mask:
                report_label = 'Aggre'
            else:
                report_label = str(lab_)
            
            if 'agg_type' in temp_true_query.columns and temp_true_query.shape[0] > 0:
                agg_type = temp_true_query.agg_type.value_counts().idxmax()
            else:
                agg_type = 'Fake'

            temp_dict = dict(
                label = report_label,
                agg_type = agg_type,
                accuracy = acc
            )
            agg_acc = pd.concat([agg_acc,pd.DataFrame(temp_dict,index=[0])],ignore_index=True)
        return agg_acc

def estimate_noise_density(noise_points, radi = .03):
    random_picks = np.random.choice(noise_points.shape[0],500)
    X_picks = noise_points[random_picks]
    tree = neighbors.KDTree(noise_points)
    test = (tree.query_radius(X_picks, r=radi, count_only=True))
    density = test/(radi**2*np.pi)
    density = density[density>np.percentile(density,25)]
    min_dens = np.median(density)
    dense_multi = 1 + np.std(density)/np.mean(density)
    return min_dens,dense_multi

def find_atleast_one_init_clust(X):
    plottle = X[:,:2]
    cords, pcs = get_contur(plottle,sigma = 1,percentile = 90,bins = 200)
    min_mask_x = plottle[:,0] > cords[:,0].min()
    min_mask_y = plottle[:,1] > cords[:,1].min()
    max_mask_x = plottle[:,0] < cords[:,0].max()
    max_mask_y = plottle[:,1] < cords[:,1].max()
    mask = min_mask_x * min_mask_y * max_mask_x * max_mask_y
    return mask
