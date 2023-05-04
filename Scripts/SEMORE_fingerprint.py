#%%
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm 
from scipy import stats,linalg,spatial,sparse,ndimage
from sklearn import neighbors,mixture,preprocessing
import cv2 as cv
import sys
from .Usefull_func import *
# =============================================================================
class Morphology_fingerprint:

    def __init__(self,data,start = 60,cut_param = 95):
        if cut_param > 1:
            cut_param = cut_param/100
        self.cut_param = cut_param
        self.frame = data.frame.astype(int)
        self.max_frame = self.frame.max()
        self.N = data.shape[0]
        if start != 'mid':
            first_frame = data.frame.min()
            data_temp = data.query('frame < @first_frame + @start').copy()
            self.start_core = np.array([data_temp.x.mean(),data_temp.y.mean()])
        if start == 'mid':
            self.start_core = np.array([data.x.values.mean(),data.y.values.mean()])
        self.y = data.y - self.start_core[-1]
        self.x = data.x - self.start_core[0]
        self.pos = np.array([self.x,self.y]).T
        self.u,self.s,self.vh = linalg.svd(self.pos)
        self.v = self.vh.T

        self.pos = np.dot(self.v,self.pos.T).T
        self.x = self.pos[:,0]
        self.y = self.pos[:,1]
        
        self.core = np.array([0,0])
        self.spat_check = False
        self.sym_check = False
        self.circ_check = False

    def sym_ratio (self):
        self.sym_check = True
        self.mask = self.pos>0
        x_dist_r, y_dist_o = np.max([[0,0],self.pos.max(axis = 0)],axis = 0)
        x_dist_l, y_dist_u = np.min([[0,0],self.pos.min(axis = 0)],axis = 0)
        self.n_r, self.n_o = self.mask.sum(axis = 0)
        self.n_l, self.n_u = self.mask.shape[0] - self.n_r, self.mask.shape[0] - self.n_o

        self.x_ratio = np.max([x_dist_r, -1*x_dist_l])/(x_dist_r+ (-1*x_dist_l))
        self.y_ratio = np.max([y_dist_o , -1*y_dist_u])/(y_dist_o+ (-1*y_dist_u))

        self.x_N_ratio = np.max([self.n_r ,self.n_l])/(self.n_r+ self.n_l)
        self.y_N_ratio = np.max([self.n_o ,self.n_u])/(self.n_o+ self.n_u)

        return_dict = {
            'x_ratio':self.x_N_ratio,
            'y_ratio':self.y_N_ratio,
            'ratio_sym': np.sqrt(self.x_N_ratio**2 + self.y_N_ratio**2),
            'x_dist_ratio':self.x_ratio,
            'y_dist_ratio':self.y_ratio
        }


        return return_dict
    
    def spatial (self,n_gauss = 4,graph_op = (False,None)):
        self.spat_check = True
        self.delon = spatial.Delaunay(self.pos)
        self.distance = np.array([spatial.distance.pdist(x) for x in self.pos[self.delon.simplices]])
        parm = stats.lognorm.fit(self.distance, floc=0)
        eval_line = np.linspace(self.distance.min(),self.distance.max(),10000)
        cdf = stats.lognorm.cdf(eval_line, *parm)
        try:
            self.ind = np.argwhere(cdf >= self.cut_param)[0][0]
            self.cut_dist = eval_line[self.ind]
        except:
            self.ind = None
            self.cut_dist = self.distance.max() + self.distance.min()
        #----------------------------------------------------------------
        cut_mask = np.product(self.distance < self.cut_dist,axis=1) == True
        self.tri_index = self.delon.simplices[cut_mask]
        self.tri_reg = self.pos[self.tri_index]
        self.area = np.sum([PArea(x[:,0],x[:,1]) for x in self.tri_reg])
        self.density = len(self.pos)/self.area
        #-----------------------------------------------------------------
        self.graph = neighbors.radius_neighbors_graph(self.pos,radius = self.cut_dist,mode = 'distance',p=2,include_self=False)
        K_bridge = self.graph.getnnz(1)
        mean_k = K_bridge.mean()
        median_k = np.median(K_bridge)
        self.min_tree = sparse.csgraph.minimum_spanning_tree(self.graph)
        distance,predecessors = sparse.csgraph.shortest_path(self.min_tree,directed=False,return_predecessors=True)

        distance[np.isinf(distance)] = -np.inf
        pred_sum = np.sum(predecessors == -9999,axis = 0)
        big_graph_index = pred_sum <= pred_sum.min()
        X_big_graph = self.pos[big_graph_index]
        OG_row = np.arange(self.pos.shape[0])[big_graph_index]
        
        hull = spatial.ConvexHull(X_big_graph)
        edge = X_big_graph[hull.vertices]
        dist_hull = spatial.distance.cdist(edge,edge)
        pair = np.unravel_index(dist_hull.argmax(), dist_hull.shape)
        indexes = tuple(hull.vertices[list(pair)])
        real_ind = OG_row[list(indexes)]
        best_pair = np.array(X_big_graph[list(indexes)])

        long_distance = distance[tuple(real_ind)]
        long_road =  np.unravel_index(distance.argmax(), distance.shape)
        
        long_road_d = spatial.distance.pdist(self.pos[list(long_road)])[0]

        long_short = get_graph_road(*real_ind,predecessors)
        long_long = get_graph_road(*long_road,predecessors)

        all_dlong = [distance[x] for x in list(zip(*long_long.T))]
        all_dshort = [distance[x] for x in list(zip(*long_short.T))]
        long_short_ratio = dist_hull.max()/long_distance
        short_long_ratio = distance.max()/long_road_d
        #----------------------------------------------------------------
        gauss_model = mixture.GaussianMixture(n_components = n_gauss,random_state = 42, n_init = 10,tol = 1e-3)
        gauss_model.fit(K_bridge.reshape(-1,1))

        sort_gauss_param = sorted(list(zip(gauss_model.means_,
                                np.sqrt(gauss_model.covariances_.ravel()),
                                gauss_model.weights_)),key = lambda x: x[0])


        graph_plot,graph_save = graph_op

        return_dict = {
            'N': self.N,
            'Area': self.area,
            'Density': self.density,
            'Cut_distance': self.cut_dist,

            'Mean_k':mean_k,
            'Median_k': median_k,
            'k_max': K_bridge.max(),

            'L_s_d': dist_hull.max(),
            'L_s_path': long_distance,
            'L_s_step': len(long_short),
            'L_s_mean': np.mean(all_dshort),
            'L_s_median': np.median(all_dshort),
            'L_l_max': np.max(all_dshort),
            'L_s_effectiveness': dist_hull.max()/len(long_short),
            'L_s_ratio': long_short_ratio,

            'L_l_d': long_road_d,
            'L_l_path': distance.max(),
            'L_l_step': len(long_long),
            'L_l_mean': np.mean(all_dlong),
            'L_l_median': np.median(all_dlong),
            'L_l_max': np.max(all_dlong),
            'L_l_effectiveness': distance.max()/len(long_long),
            'L_l_ratio': short_long_ratio,

            'Ls_ll_d_ratio': dist_hull.max()/long_road_d,
            'Ls_ll_path_ratio': long_distance/distance.max(),

            'State_diff': np.diff(np.array(sort_gauss_param,dtype = object)[:,0]).mean()
        }
        a = 1
        for mu,sig,w in sort_gauss_param:
            return_dict['mu_{}'.format(a)] = mu
            return_dict['sig_{}'.format(a)] = sig
            return_dict['w_{}'.format(a)] = w
            a+=1
        
        if graph_plot:
            min_graph = get_graph_cords(self.pos,self.min_tree)
            fig = plt.figure(dpi = 150,figsize = (20,25))
            n_rows = 3
            ax_big = fig.add_subplot(n_rows,1,1)
            ax_big.scatter(*self.pos.T,s = 5,c = self.graph.getnnz(1),cmap = 'inferno',zorder = 2)
            _ = ax_big.plot(*min_graph,color = 'r',lw = 2,zorder = 1)
            line_short = ax_big.plot(*best_pair.T,ls = '--',lw = 2)
            road_short = ax_big.plot(*self.pos[long_short].T,lw = 2,zorder = 5,color = 'green',alpha = .5)
            road_long  = ax_big.plot(*self.pos[long_long].T,lw = 2,zorder =4,color = 'k',alpha =.8)
            line_long  = ax_big.plot(*self.pos[list(long_road)].T,ls = '--', c='orange',lw = 2)
            ax_big.triplot(*self.pos.T,self.tri_index,zorder =1,alpha=0.1,color = 'k')
            road_short[0].set_label('{:.0f}'.format(long_distance))
            line_short[0].set_label('{:.0f}'.format(dist_hull.max()))
            road_long[0].set_label('{:.0f}'.format(distance.max()))
            line_long[0].set_label('{:.0f}'.format(spatial.distance.pdist(self.pos[list(long_road)])[0]))

            ax_big.legend(ncol = 2, loc = 'upper right',fontsize = 12)
            ax = fig.add_subplot(n_rows,3,4)
            ax1 = fig.add_subplot(n_rows,3,5)
            ax2 = fig.add_subplot(n_rows,3,6)

            ax_grp = fig.add_subplot(n_rows,2,5)
            ax_grp_prop = fig.add_subplot(n_rows,2,6)


            ax.hist(self.graph.getnnz(1),bins = 20,density = True,histtype = 'step',color = 'k',label = 'N |{:.0f}'.format(self.graph.getnnz(1).shape[0]))
            x_plot = np.linspace(self.graph.getnnz(1).min(),self.graph.getnnz(1).max(),1000)
            collective_y = np.zeros(x_plot.shape[0])
            color_map = cm.get_cmap('gist_rainbow',gauss_model.n_components)
            a= 0
            for mu,sig,w in sorted(list(zip(gauss_model.means_,
                                            np.sqrt(gauss_model.covariances_.ravel()),
                                            gauss_model.weights_)),key = lambda x: x[0]):
                y = w * stats.norm.pdf(x_plot, mu, sig)
                collective_y +=y
                ax.plot(x_plot,y,lw = 2,c = color_map(a),label = 'State | {}'.format(a))
                a+=1
            ax.plot(x_plot,collective_y,lw = 1,color = 'purple')
            ax.legend()
            arr = np.array(sorted(list(zip(gauss_model.means_.ravel(),
                                        range(n_gauss))),
                                key = lambda x: x[0]),
                        dtype = object)
            true_pred = dict(zip(arr[:,1],range(n_gauss)))
            prediction = np.array([true_pred[x] for x in gauss_model.predict(self.graph.getnnz(1).reshape(-1,1))])
            prediction_proba = prediction + (1 - gauss_model.predict_proba(self.graph.getnnz(1).reshape(-1,1)).max(axis = 1))
            ax.set_title('N_connection')
            ax1.hist(all_dlong,bins = 50)
            ax1.set_title('Distances_long')
            ax2.hist(all_dshort,bins = 50)
            ax2.set_title('Distances_short')
            ax_grp.scatter(*self.pos.T,s = 5,c = prediction, cmap = 'gist_rainbow', vmax = prediction.max())
            propa = ax_grp_prop.scatter(*self.pos.T,s = 5,c = prediction_proba ,cmap = 'gist_rainbow',vmax = prediction.max())
            ax_grp.set_title('Prediction for GaussMixture')
            ax_grp_prop.set_title('Prediction + Probability for GaussMixture')
            fig.colorbar(propa,ax = ax_grp_prop)
            if graph_save is not None:
                fig.savefig('graph_save.png')

        return return_dict
    
    def circularity (self,sigma = 2,percentile = 80,bins = 150, weights = True):
        self.circ_check = True
        circ_scaler = preprocessing.StandardScaler().fit(self.pos)
        if weights is False:
            weights = np.ones(self.pos.shape[0])
        else:
            try: 
                weights = self.graph.getnnz(1)
            except:
                self.spatial()
                weights = self.graph.getnnz(1)
        contour,pcs = get_contur(self.pos,sigma = sigma,percentile = percentile,bins = bins,weights = weights)
        contour = contour.astype('float32')
        self.cont = contour
        cont_length = cv.arcLength(contour,True)
        cont_area = cv.contourArea(contour)
        circularity = 4*np.pi*cont_area/cont_length**2
        convexity = cont_area/cv.contourArea(cv.convexHull(contour))
        inertia = np.divide(*cv.fitEllipse(contour)[1])
        
        # Variance
        test = circ_scaler.transform(contour) - (self.core - circ_scaler.mean_ )/np.sqrt(circ_scaler.var_)
        var_mid = np.sqrt(np.sum(test**2,axis = 1)).var()
        Sph_value = np.mean([(1-var_mid),circularity,convexity,inertia])

        # Add it up
        return_dict ={
            'Variance':var_mid,
            'Circularity': circularity,
            'Convexity': convexity,
            'Circ_inertia': inertia,
            'Sph_value': Sph_value
            } 
        return return_dict
    
    def correlation (self):
        self.corr_check = True
        self.pear,pear_p = stats.pearsonr(self.x,self.y)
        self.spear,spear_p = stats.spearmanr(self.x,self.y)

        return_dict = {
            'Pearson': self.pear,
            'Spearman': self.spear
        }
        return return_dict

    def all_run(self,spatial_param = {},circ_param = {}):
        sym = self.sym_ratio()
        spat= self.spatial(**spatial_param)
        circ = self.circularity(**circ_param)
        corr = self.correlation()
        return_dict = {**sym, **spat, **circ, **corr}
        return return_dict

    def plot_features (self,
                    title = False, 
                    ax = None, 
                    graph = False,
                    triangle = False,
                    circ = False,
                    sym = False,
                    verbose = False,
                    cmap = 'inferno',
                    **kwargs
                    ):
        if (graph or triangle) and self.spat_check is not True: 
            _=self.spatial()
        if sym and self.sym_check is not True: 
            _=self.sym_ratio()
        if circ and self.circ_check is not True: 
            _ = self.circularity()
        if ax == None:
            fig,ax = plt.subplots(dpi=300,figsize=(12,8))
        if self.spat_check: 
            color_type = self.graph.getnnz(1)
        else:
            color_type = self.frame
        ax.scatter(self.x,self.y,c=color_type,s = 10,cmap = cmap,zorder = 10)
        ax.set_xlim(self.x.min()*1.1,self.x.max()*1.1)
        ax.set_ylim(self.y.min()*1.1,self.y.max()*1.1)

        if triangle:
            ax.triplot(self.x,self.y,self.tri_index,zorder =1,alpha=1)
        if graph:
            graph_cords = get_graph_cords(self.pos,self.min_tree)
            #big_grapg = get_graph_cords(self.pos,self.graph)
            ax.plot(*graph_cords,c = 'r',zorder = 1)
            #ax.plot(*big_grapg,c = 'k',zorder = 0,alpha = .5)
            ax.plot()
        if circ:
            ax.plot(*self.cont.T,c = 'aqua',lw = 5,zorder = 11)
            ax.scatter(*self.core,c = 'aqua',s = 150,marker = '*',zorder = 11)
        if sym:
            text_lowerright =    'N | {}\nRatio |  {:.2f}'.format(np.sum(self.mask[:,0] * ~self.mask[:,1]),np.sum(self.mask[:,0] * ~self.mask[:,1])/len(self.x))
            text_lowerleft =   'N | {}\nRatio |  {:.2f}'.format(np.product(~self.mask,axis = 1).sum(),np.product(~self.mask,axis = 1).sum()/len(self.x))
            text_topleft = 'N | {}\nRatio |  {:.2f}'.format(np.sum(~self.mask[:,0] * self.mask[:,1]),np.sum(~self.mask[:,0] * self.mask[:,1])/len(self.x))
            text_topright =  'N | {}\nRatio |  {:.2f}'.format(np.product(self.mask,axis = 1).sum(),np.product(self.mask,axis = 1).sum()/len(self.x))

            ax.hlines(0,self.x.min()*1.1,self.x.max()*1.1,color='k',ls='--')
            ax.vlines(0,self.y.min()*1.1,self.y.max()*1.1,color='k',ls='--')
            ax.fill_between(np.linspace(0,self.x.min()*1.1,1000),np.repeat(self.y.max()*1.1,1000),alpha = .3,zorder = 0,color ='k',interpolate = True)
            ax.fill_between(np.linspace(0,self.x.max()*1.1,1000),np.repeat(self.y.max()*1.1,1000),alpha = .3,zorder = 0,color ='green',interpolate = True)
            ax.fill_between(np.linspace(0,self.x.min()*1.1,1000),np.repeat(self.y.min()*1.1,1000),alpha = .3,zorder = 0,color ='orange',interpolate = True)    
            ax.fill_between(np.linspace(0,self.x.max()*1.1,1000),np.repeat(self.y.min()*1.1,1000),alpha = .3,zorder = 0,color ='red',interpolate = True)
            ax.text(0.05, 0.9, text_topleft, horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round",alpha = .5,fc = 'white',ec = 'k'
                                ))
            ax.text(0.9, 0.9, text_topright, horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round",alpha = .5,fc = 'white',ec = 'k'
                                ))
            ax.text(0.05, 0.1, text_lowerleft, horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round",alpha = .5,fc = 'white',ec = 'k'
                                ))
            ax.text(0.9, 0.1, text_lowerright, horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(boxstyle="round",alpha = .5,fc = 'white',ec = 'k'
                                ))
        if verbose:
            ax.plot(self.best_pair[:,0],self.best_pair[:,1],ls = '--',lw = 3,c='white',alpha = 1,label = 'Longest dist',zorder=4)
            ax.add_patch(Polygon(self.edge,alpha = 0.3,zorder=1,ls='-',lw=5,edgecolor='k',label ='Covexhull'))

        if title != False:
            ax.set_title(str(title))
        
        ax.set(**kwargs)
        
#%%
if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('/Users/steenbender/Desktop/Master_Thesis/Spherulite project/Output/control/Found_clusters/csv_files/Cluster_17.csv')
    #fig = plt.figure(figsize=(12,8),dpi = 150)
    #plt.scatter(data.x,data.y)
    clust = Sph_features(data)
    feat = clust.all_run()
    clust.plot_sphere(graph = False,triangle = False,circ = True,sym = False)
    #_ = clust.spatial(graph_op = (True,None))
# %%
