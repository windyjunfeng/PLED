# coding=utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt


metric_representation={'mae':'MAE (V/m)','mre':'MRE','mse':'MSE','f_norm':'F-Norm Loss','psnr':'PSNR (dB)','pearson_corr':'PCC','mae_loss_mean':'MAE of Mean','mae_loss_edge':'AAE (V/m)','mre_loss_edge':'ARE','vector_angle':'Directional deviation (°)','vector_diff_magnitude':'Magnitude of the vector difference (V/m)'}
classnames={'class_1':'WM','class_2':'GM','class_3':'CSF','class_4':'Bone','class_5':'Scalp','class_9':'Blood'}
# methods_folders={'Origin':'35_s','Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}
# methods_folders={'based_step1':'39_final_new_based_step1','based_gt':'39_final_new_based_gtlabel'}
methods_folders={'Origin':'45_efield_vector','Baseline':'46_based_step1_efield_vector','PLED':'42_4_based_step1_efield_vector'}

# input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
# output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis'
input_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health'
output_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/results_analysis'
# sub_foldername=r'box_plot_viridis_nolegend\total'
# sub_foldername=r'box_plot_viridis_nolegend_39_based_gt_step1\total'
sub_foldername=r'efield_vector_box_plot_viridis_nolegend_pvalue/total'
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
    # import pdb
    # pdb.set_trace()
    # data['Tissues']=data['Tissues'].astype('category')
    # data['Methods'] = data['Methods'].astype('category')
    p_values=[]
    for classname in classnames.keys():
        class_data=data[data['Tissues']==classnames[classname]]
        pairwise_results = pg.pairwise_tests(dv=metric_representation[key],between='Methods', data=class_data, parametric=True,padjust='bonf')  # T检验，Bonferroni correction，参数化检验，pairwise_results.columns可以查看结果字典中的key有哪些
        p_values.append([pairwise_results['p-corr'][2],pairwise_results['p-corr'][1]])
    # ymax=0
    # for artist in ax.artists:
    #     upper_whisker = artist.get_ydata()[3]
    #     if upper_whisker>ymax:
    #         ymax=upper_whisker
    # ymax = ymax + 1
    # import pdb
    # pdb.set_trace()
    ymin,ymax=ax.get_ylim()
    y_range=ymax-ymin
    for i in range(6):
        for j in range(2):
            if j==0:
                width_left=0.4
                width_right=0.4
                height_line=0.05
                height_star=0.035
                p_val=p_values[i][j]
            else:
                width_left = 0.1
                width_right = 0.4
                height_line = 0
                height_star = -0.015  # 由于*并不是贴着底线写的，因此需要向下平移
                p_val = p_values[i][j]
            # 绘制横线
            plt.plot([i - width_left, i + width_right], [ymax+height_line*y_range, ymax+height_line*y_range], color='black', lw=1.5)
            # 添加星号表示显著性
            if p_val < 0.05 and p_val >= 0.01:
                plt.text(i+(width_right-width_left)/2, ymax + height_star*y_range, f"*", ha='center', va='bottom', fontsize=12)
            elif p_val<0.01 and p_val >=0.001:
                plt.text(i+(width_right-width_left)/2, ymax + height_star*y_range, f"**", ha='center', va='bottom', fontsize=12)
            elif p_val<0.001:
                plt.text(i+(width_right-width_left)/2, ymax + height_star*y_range, f"***", ha='center', va='bottom', fontsize=12)
    plt.savefig(os.path.join(output_path,key+'_category_pvalue.png'),dpi=1200,bbox_inches='tight', pad_inches=0.1, format='png')
