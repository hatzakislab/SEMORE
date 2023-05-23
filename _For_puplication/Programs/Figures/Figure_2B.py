"""
This scripts creates the presented results and heatmap seen in Figure 2B of the article.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import pandas as pd
import sys
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
sys.path.append('../../../')
from Scripts.Usefull_func import fix_labels, find_acc

sim_type = ['Iso_sim','Rand_sim','Fib_sim']
for sim in sim_type:
    path = f'../../Output/Stdr_sim/{sim}/big_csv'
    files = glob(path+'/*.csv')
    performance_dataframe = pd.DataFrame()
    for file in tqdm(files):
        temp_data = pd.read_csv(file)
        temp_data.Final_label = fix_labels(temp_data.Final_label, temp_data.label).astype(int)
        temp_acc = find_acc(temp_data,true_col='label',pred_col='Final_label')
        median,mean,var,noise = temp_acc.accuracy.median(numeric_only = True),temp_acc.accuracy.mean(numeric_only = True),temp_acc.accuracy.var(numeric_only = True),temp_acc.query('label == "Noise"').accuracy.values
        F1, Precision, Recall = precision_recall_fscore_support(temp_data.label,temp_data.Final_label,average='micro',zero_division= 0)[:-1]
        TP,FP,TN,FN = temp_data.query('Final_label == label & label != -1').shape[0],\
                      temp_data.query('Final_label != label & label == -1').shape[0],\
                      temp_data.query('Final_label == label & label == -1').shape[0],\
                      temp_data.query('Final_label != label & Final_label == -1').shape[0]
       
        TP_scaled = TP/(TP+FN)
        FN_scaled = FN/(TP+FN)
        FP_scaled = FP/(FP+TN) 
        TN_scaled = TN/(TN+FP)
        performance_dict = dict(
            median = median,
            mean = mean,
            var = var,
            noise = noise,
            F1 = F1,
            Precision = Precision,
            Recall = Recall,
            TP = TP_scaled,
            FP = FP_scaled,
            TN = TN_scaled,
            FN = FN_scaled,
        )
        performance_dataframe = pd.concat([performance_dataframe,pd.DataFrame(performance_dict,index = [file.split('/')[-1].split('.')[0]])])

        
    figure = plt.figure(figsize=(10,10))
    ax = figure.add_subplot(111)
    cm = performance_dataframe[['TP','FN','FP','TN']].mean().values.reshape(2,2) 
    cm_std = performance_dataframe[['TP','FN','FP','TN']].std().values.reshape(2,2)
    print('--------')
    print(cm_std)
    print('--------')
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        linewidths=5,
        linecolor = 'k',
        square = True,
        cmap = 'Reds',
        cbar = False,
        ax = ax,
        vmin = 0,
        vmax = 1,
        annot_kws={"size": 54},
        xticklabels = ['ID','Noise'],
        yticklabels = ['ID','Noise'],

    )
    ax.set_xlabel('Predicted label',fontsize = 64)
    ax.set_ylabel('True label',fontsize = 64)
    ax.tick_params(axis='both', which='major', labelsize=54)
    plt.show()
    print(f'{sim} sim')
    print('--------')
    print(performance_dataframe[['median','mean','var','noise','F1','Precision','Recall']].agg(['mean','median','std']))
    print('--------')