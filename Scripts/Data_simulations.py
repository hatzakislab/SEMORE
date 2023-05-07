import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import stats, spatial

def gen_iso (df_slice:pd.DataFrame) -> pd.DataFrame:
    """
    Generate a dataframe of simulated isotropic aggregates.

    Parameters
    ----------
    df_slice : pd.DataFrame
        A dataframe with the following columns:
            'seed': The starting point of the aggregate
            'start': The starting frame of the aggregate
            'end': The ending frame of the aggregate

    Returns
    -------
    pd.DataFrame
        A dataframe with the following columns
            'frame': The frame of the aggregate
            'x': The x coordinate of the aggregate
            'y': The y coordinate of the aggregate
            'label': The label of the aggregate
            'agg_type': The type of aggregation
    """
    end_df = pd.DataFrame()
    lab = 0
    for seed,start,end in zip(df_slice['seed'],df_slice['start'],df_slice['end']):
        all_points = {'frame': [start],
                    'x': [seed[0]],
                    'y': [seed[1]],
                    'label': [],
                    'agg_type': []}
        
        for i in range(start,end):
            pos = np.array([all_points['x'],all_points['y']]).T
            kde = KernelDensity(kernel='gaussian', bandwidth=((i-start)/(end-start)*end+1)*10).fit(pos)
            new_n = np.random.randint(0, 30)
            new_point = kde.sample(n_samples = new_n)

            all_points['x'].extend(new_point[:, 0])
            all_points['y'].extend(new_point[:, 1])
            all_points['frame'].extend([i+1]*new_n)
        all_points['label'].extend([lab]*len(all_points['frame']))
        all_points['agg_type'].extend(['Iso']*len(all_points['frame']))
        lab += 1
        end_df = pd.concat([end_df,pd.DataFrame(all_points)])
    return end_df

def gen_rand (df_slice:pd.DataFrame,lam = 10, threshold = np.inf,sigma = 500) -> pd.DataFrame:
    """
    Generate a dataframe of simulated random aggregates throguh sterically hinderince.

    Parameters
    ----------
    df_slice : pd.DataFrame
        A dataframe with the following columns:
            'seed': The starting point of the aggregate
            'start': The starting frame of the aggregate
            'end': The ending frame of the aggregate
    lam : int, optional
        The lambda value for the poisson distribution used to determine the number of points added at each frame, by default 10
    threshold : float, optional
        The maximum distance between points for it to be considered. by default np.inf
    sigma : int, optional
        The standard deviation of the normal distribution used to determine the new points, by default 500

    Returns
    -------
    pd.DataFrame
        A dataframe with the following columns
            'frame': The frame of the aggregate
            'x': The x coordinate of the aggregate
            'y': The y coordinate of the aggregate
            'label': The label of the aggregate
            'agg_type': The type of aggregation
    """
    end_df = pd.DataFrame()
    lab = 0
    for seed,start,end in zip(df_slice['seed'],df_slice['start'],df_slice['end']):
        current_points = np.asarray(seed).reshape(-1,2)
        frame_list = [start]
        for i in range(start,end):
            n_point_add = np.min([len(current_points),np.random.poisson(lam = lam)])
            if n_point_add == 0:
                continue
            new_points = current_points[-50:] + np.random.normal(0, sigma,size = (current_points[-50:].shape[0],2))
            distances = spatial.distance.cdist(current_points[-50:], new_points)
            distances[distances > threshold] = np.nan
            prop_value = np.nansum(np.exp(-distances**2),axis = 1)
            argmin = np.argsort(prop_value)[:n_point_add]
            current_points = np.append(current_points, new_points[argmin], axis=0)
            frame_list.extend([i]*np.ones(len(argmin)))

        return_df = {
            'frame':frame_list,
            'x':current_points[:,0],
            'y':current_points[:,1],
            'label': [lab] * (len(current_points)),
            'agg_type': ['Rand'] * (len(current_points))
        }
        end_df = pd.concat([end_df,pd.DataFrame(return_df)])
        lab += 1
    return end_df

def gen_fib (df_slice:pd.DataFrame,max_branch = 3,branch_p = 0.005,lambda_dock = 1,dock_min = 0,min_size = 100) -> pd.DataFrame:
    """
    Generate a dataframe of simulated fibrils

    Parameters
    ----------
    df_slice : pd.DataFrame
        A dataframe with the following columns:
            'seed': The starting point of the fibril
            'start': The starting frame of the fibril
            'end': The ending frame of the fibril
    max_branch : int, optional
        The maximum number of branches a fibril can have, by default 3
    branch_p : float, optional
        The probability of a branch occuring at each frame, by default 0.005
    lambda_dock : int, optional
        The lambda value for the poisson distribution used to determine the number of docks, by default 1
    dock_min : int, optional
        The minimum number of docks, by default 0
    min_size : int, optional
        The minimum size of a fibril, by default 100

    Returns
    -------
    pd.DataFrame
        A dataframe with the following columns
            'frame': The frame of the fibril
            'x': The x coordinate of the fibril
            'y': The y coordinate of the fibril
            'Fibril_label': The label of the fibril
            'agg_type': The type of aggregation
            'N_branch': The number of branches
    """
    n_docks = lambda min_: np.max([min_,np.random.poisson(lam = lambda_dock)]).astype(int)
    return_df = pd.DataFrame()
    a = 0
    for seed,start,end in zip(df_slice['seed'],df_slice['start'],df_slice['end']):
        radi_start = np.random.uniform(0,2*np.pi)
        final_dict = {
            'frame': [start],
            'x': [seed[0]],
            'y': [seed[1]],
            'label': [0],
            'agg_type': ['Fib'],
            'radi': [0],
            'direction':[radi_start]
        }
        final_dict = pd.DataFrame(final_dict)
        n_branch = 1
        for i in range(start,end+1):
            for branch in np.unique(final_dict['label']):
                current_branch = final_dict.query('label == @branch')
                check_branch = np.random.uniform(0,1)
                if check_branch < branch_p and n_branch < max_branch:
                    n_point = n_docks(1)
                    n_branch += 1
                    new_branch_pos = current_branch.iloc[-1][['x','y']].values
                    new_branch_frame = i
                    branch_direction = np.random.choice([1,-1])
                    new_branch_radi = np.random.normal(
                        loc = branch_direction * np.pi/4 + current_branch.direction.values[-1],
                        scale = np.pi/16,
                        size = n_point
                    )
                    new_branch_label = np.max(final_dict['label']) + 1
                    step_len = np.random.normal(loc = 100,scale = 20,size = n_point)
                    new_branch_dict = {
                        'radi': new_branch_radi,
                        'direction': [new_branch_radi[0]]*n_point,
                        'x': new_branch_pos[0] + step_len * np.cos(new_branch_radi),
                        'y': new_branch_pos[1] + step_len * np.sin(new_branch_radi),
                        'frame': new_branch_frame * np.ones(n_point),
                        'label': new_branch_label * np.ones(n_point),
                        'agg_type': ['Fib'] * n_point
                    }
                    final_dict = pd.concat([
                        final_dict,
                        pd.DataFrame(new_branch_dict)],
                        ignore_index = True
                        )
                    current_branch = final_dict.query('label == @branch')
                n_point = n_docks(dock_min)
                step_len = np.random.normal(loc = 100,scale = 20,size = n_point)
                if len(final_dict.radi)>=2:
                    radi_temp = current_branch.direction.values[-1]
                    radi_next = stats.norm.rvs(loc = radi_temp,scale = np.pi/4,size = n_point)

                    temp_dict = {
                    'radi': radi_next,
                    'direction': radi_temp * np.ones(n_point),
                    'x': current_branch.x[current_branch.frame == current_branch.frame.values[-1]].values.mean() + step_len * np.cos(radi_next),
                    'y': current_branch.y[current_branch.frame == current_branch.frame.values[-1]].values.mean() + step_len * np.sin(radi_next),
                    'frame': i * np.ones(n_point),
                    'label': branch * np.ones(n_point),
                    'agg_type': ['Fib'] * n_point
                    }

                else:
                    radi_temp = final_dict['direction']
                    radi_next = stats.norm.rvs(loc = radi_temp,scale = np.pi/4,size = n_point)

                    temp_dict = {
                        'radi': radi_next,
                        'direction': list(radi_temp) * n_point,
                        'x': final_dict['x'].values[0] + step_len * np.cos(radi_next),
                        'y': final_dict['y'].values[0] + step_len * np.sin(radi_next),
                        'frame': i,
                        'label': branch,
                        'agg_type': ['Fib'] * n_point
                    }
                final_dict = pd.concat([final_dict , pd.DataFrame(temp_dict,index = np.arange(len(temp_dict['radi'])))],ignore_index = True)
        final_dict['Fibril_label'] = [a]*len(final_dict)
        final_dict['N_branch'] = [n_branch] * len(final_dict) 
        return_df = pd.concat([return_df,final_dict],ignore_index = True)
        a+=1
    return_df = return_df[['frame','x','y','Fibril_label','agg_type','N_branch']]
    return_df.rename(columns = {'Fibril_label':'label'},inplace = True)
    return return_df


def agg_sim (
        n_agg :int = 1, 
        noise_ratio:float = 0.0,
        max_frame:int = 400,
        min_duration:int = 100, 
        fov: int = 40000,
        include_ratio:np.array = np.array([.34, .33,.33]),
        start_seeds: np.array = None ) -> pd.DataFrame:
    """
    Simulate aggregation with different types of aggregation
    Parameters
    ----------
    n_agg : int, optional
        number of aggregation to simulate, by default 1
    noise_ratio : float, optional
        ratio of noise to add, by default 0.0
    max_frame : int, optional 
        maximum number of frame, by default 400
    min_duration : int, optional
        minimum duration of aggregation, by default 100
    fov : int, optional
        field of view, by default 40000
    include_ratio : np.array, optional
        ratio of different types of aggregation, by default np.array([.34, .33,.33])
    start_seeds : np.array, optional
        starting seeds, by default None
    Returns
    -------
    pd.DataFrame
        Containing the simulated aggregation with columns ['frame','x','y','label','agg_type', 'N_branch']
            'frame' : frame number
            'x' : x coordinate
            'y' : y coordinate
            'label' : label of the aggregation
            'agg_type' : type of aggregation
            'N_branch' : number of branches
    """
    
    if start_seeds is not None:
        seeds = np.asanyarray(start_seeds)
    else:
        seeds  = np.random.uniform(-fov,fov,size = (n_agg,2))

    seed_selection = np.random.choice(['Iso','Rand','Fib'], size = seeds.shape[0], p = include_ratio)
    frame_start = np.random.randint(0, max_frame-min_duration, size = seeds.shape[0])
    frame_end = np.random.randint(frame_start+min_duration, max_frame, size = seeds.shape[0])
    sorted_seed = pd.DataFrame(dict(agg_type = seed_selection, seed = list(seeds),start = frame_start, end = frame_end))
    iso_agg, rand_agg, fib_agg = None, None, None
    if 'Iso' in seed_selection:
        iso_agg = gen_iso(sorted_seed.query('agg_type == "Iso"'))
    if 'Rand' in seed_selection:
        rand_agg = gen_rand(sorted_seed.query('agg_type == "Rand"'))
    if 'Fib' in seed_selection:
        fib_agg = gen_fib(sorted_seed.query('agg_type == "Fib"'))

    agg_df = pd.concat([iso_agg,rand_agg,fib_agg],ignore_index = True)

    noise_df = pd.DataFrame(dict(
        x = np.random.uniform(-fov,fov,size = int(len(agg_df)*noise_ratio)),
        y = np.random.uniform(-fov,fov,size = int(len(agg_df)*noise_ratio)),
        frame = np.random.randint(0,max_frame,size = int(len(agg_df)*noise_ratio)),
        label = -1 * np.ones(int(len(agg_df)*noise_ratio)),
        agg_type = ['Noise']*int(len(agg_df)*noise_ratio)
    ))
    agg_df = pd.concat([agg_df,noise_df],ignore_index = True)
    agg_cut = agg_df.query('x >= -@fov & x <= @fov & y >= -@fov & y <= @fov')


    return agg_cut