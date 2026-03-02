# coding=utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# subjects=['m2m_A2','m2m_A3','m2m_A8','m2m_A9','m2m_A14','m2m_A17','m2m_A18','m2m_A20','m2m_A24']
# methods_folders={'Origin':'35','Baseline':'31_based_step1','PLED':'39_based_step1'}
# subjects=['m2m_390645','m2m_432332','m2m_566454','m2m_644044','m2m_753150']
subjects=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
methods_folders={'Origin':'35_cog','Baseline':'31_based_step1_cog','PLED':'39_based_step1_cog'}
classnames={'gm':'WM','foreground':'Total'}
aspects={'pos':'Distance of optimal coil positions (mm)','angle':'MAE of optimal coil orientation (°)','efield':'MRE of optimal E-field magnitude (%)'}
input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\optimization_evaluation'
output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\optimization_evaluation\visualization_cog'
# sub_foldername=r'box_plot_viridis_nolegend\total'
# sub_foldername=r'box_plot_viridis_nolegend_39_based_gt_step1\total'
# gt_foldername=r'gt'
gt_foldername=r'gt_cog'
sub_foldername=r'violin_plot_viridis_pvalue'
output_path=os.path.join(output_root_path,sub_foldername)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def append_value(total,item):
    if total is None:
        total=item.copy()
    else:
        total=np.concatenate((total,item))
    return total


# 定义过滤异常点的函数
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


for classname in classnames.keys():
    values=None
    categories=None
    methods=None
    for aspect in aspects.keys():
        for method in methods_folders.keys():
            for subject in subjects:

                # if method=='Origin' and subject not in ['m2m_A2','m2m_A9']:
                #     continue

                gt_path=os.path.join(input_root_path,gt_foldername,'optimal_eval',subject.split('_')[-1]+'_'+classname+'_best_'+aspect+'.npy')
                gt_data=np.load(gt_path)
                predict_path=os.path.join(input_root_path,methods_folders[method],'optimal_eval',subject.split('_')[-1]+'_'+classname+'_best_'+aspect+'.npy')
                predict_data=np.load(predict_path)
                if aspect=='efiled':
                    eval=np.abs(predict_data-gt_data)/gt_data  # MRE
                elif aspect=='pos':
                    eval=np.linalg.norm((predict_data-gt_data),axis=1)  # l2距离
                else:
                    eval=np.abs(predict_data-gt_data) # MAE
                mask=np.isfinite(eval)
                values_filtered=eval[mask]  # 过滤掉为nan或inf的元素
                method_item=np.array([method]*len(values_filtered))
                values=append_value(values,values_filtered)
                methods=append_value(methods,method_item)
        data = pd.DataFrame({'Methods': methods, aspects[aspect]: values})
        # 过滤异常点
        data_filtered = data.groupby("Methods").apply(lambda x: remove_outliers(x, aspects[aspect])).reset_index(drop=True)
        plt.figure()
        ax=sns.violinplot(data=data_filtered, x='Methods', y=aspects[aspect], hue='Methods', linewidth=1.5, inner=None, palette='viridis')  # 不显示离群点
        # ax.legend_.remove()
        output_class_path=os.path.join(output_path,classname)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        plt.savefig(os.path.join(output_class_path,aspect+'_optimization.png'),dpi=1200,bbox_inches='tight', pad_inches=0.1, format='png')
