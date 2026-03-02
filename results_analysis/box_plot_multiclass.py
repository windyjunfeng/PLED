# coding=utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


metric_representation={'mae':'MAE','mre':'MRE','mse':'MSE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mae_loss_mean':'MAE of Mean','mae_loss_edge':'AAE','mre_loss_edge':'ARE'}
keys_list = list(metric_representation.keys())
classnames={'class_1':'WM','class_2':'GM','class_3':'CSF','class_4':'Bone','class_5':'Scalp','class_9':'Blood'}
root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\35_s'
sub_foldername=r'box_plot_plasma'
output_path=os.path.join(root_path,sub_foldername)
if not os.path.exists(output_path):
    os.makedirs(output_path)
for i in range(len(keys_list)):
    # path_wm=os.path.join(root_path,classnames[0],keys_list[i]+'_'+classnames[0]+'.npy')
    # path_gm=os.path.join(root_path,classnames[1],keys_list[i]+'_'+classnames[1]+'.npy')
    # path_csf=os.path.join(root_path,classnames[2],keys_list[i]+'_'+classnames[2]+'.npy')
    # path_bone=os.path.join(root_path,classnames[3],keys_list[i]+'_'+classnames[3]+'.npy')
    # path_scalp=os.path.join(root_path,classnames[4],keys_list[i]+'_'+classnames[4]+'.npy')
    # path_blood=os.path.join(root_path,classnames[5],keys_list[i]+'_'+classnames[5]+'.npy')
    # values_wm=np.load(path_wm)
    # values_gm=np.load(path_gm)
    # values_csf=np.load(path_csf)
    # values_bone=np.load(path_bone)
    # values_scalp=np.load(path_scalp)
    # values_blood=np.load(path_blood)
    # categories = np.array(['WM'] * len(values_wm) + ['GM'] * len(values_gm) + ['CSF'] * len(values_csf) + ['Bone'] * len(values_bone) + ['Scalp'] * len(values_scalp) + ['Blood'] * len(values_blood))
    # values = np.concatenate([values_wm, values_gm, values_csf, values_bone, values_scalp, values_blood])
    values=None
    categories=None
    for classname in classnames.keys():
        path=os.path.join(root_path,classname,keys_list[i]+'_'+classname+'.npy')
        values_origin=np.load(path)
        mask=np.isfinite(values_origin)
        values_filtered=values_origin[mask]  # 过滤掉为nan或inf的元素
        category_item= np.array([classnames[classname]] * len(values_filtered))
        if categories is None:
            categories = category_item
        else:
            categories = np.concatenate([categories,category_item])
        if values is None:
            values=values_filtered
        else:
            values=np.concatenate([values,values_filtered])

    data = pd.DataFrame({
        'Tissues': categories,
        metric_representation[keys_list[i]]: values
    })
    plt.figure()
    sns.boxplot(data=data, x='Tissues', y=metric_representation[keys_list[i]], linewidth=1.5,showfliers=False,palette='plasma')  # 不显示离群点
    # plt.title('')
    # plt.xlabel('Columns')
    # plt.ylabel('Values')
    # plt.show()
    plt.savefig(os.path.join(output_path,metric_representation[keys_list[i]]+'_category.png'),dpi=1200,bbox_inches='tight', pad_inches=0.1, format='png')