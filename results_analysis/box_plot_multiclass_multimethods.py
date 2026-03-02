# coding=utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


metric_representation={'mae':'MAE','mre':'MRE','mse':'MSE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mae_loss_mean':'MAE of Mean','mae_loss_edge':'AAE','mre_loss_edge':'ARE'}
classnames={'class_1':'WM','class_2':'GM','class_3':'CSF','class_4':'Bone','class_5':'Scalp','class_9':'Blood'}
# methods_folders={'Origin':'35_s','Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}
methods_folders={'based_step1':'39_final_new_based_step1','based_gt':'39_final_new_based_gtlabel'}

input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis'
# sub_foldername=r'box_plot_viridis_nolegend\total'
sub_foldername=r'box_plot_viridis_nolegend_39_based_gt_step1\total'
output_path=os.path.join(output_root_path,sub_foldername)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def append_value(total,item):
    if total is None:
        total=item.copy()
    else:
        total=np.concatenate((total,item))
    return total


for key in metric_representation.keys():
    values=None
    categories=None
    methods=None
    for method in methods_folders.keys():
        for classname in classnames.keys():
            path=os.path.join(os.path.join(input_root_path,methods_folders[method]),classname,key+'_'+classname+'.npy')
            values_origin=np.load(path)
            mask=np.isfinite(values_origin)
            values_filtered=values_origin[mask]  # 过滤掉为nan或inf的元素
            category_item= np.array([classnames[classname]] * len(values_filtered))
            method_item=np.array([method]*len(values_filtered))
            categories=append_value(categories,category_item)
            values=append_value(values,values_filtered)
            methods=append_value(methods,method_item)
    data = pd.DataFrame({
        'Methods': methods,
        'Tissues': categories,
        metric_representation[key]: values
    })
    plt.figure()
    # sns.boxplot(data=data, x='Tissues', y=metric_representation[key], hue='Methods',linewidth=1.5,showfliers=False,palette='viridis')  # 不显示离群点
    ax=sns.boxplot(data=data, x='Tissues', y=metric_representation[key], hue='Methods', linewidth=1.5, showfliers=False, palette='viridis')  # 不显示离群点
    ax.legend_.remove()
    plt.savefig(os.path.join(output_path,metric_representation[key]+'_category.png'),dpi=1200,bbox_inches='tight', pad_inches=0.1, format='png')
