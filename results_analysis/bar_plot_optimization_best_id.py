# coding=utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size':15})

subjects=['m2m_A2','m2m_A3','m2m_A8','m2m_A9','m2m_A14','m2m_A17','m2m_A18','m2m_A20','m2m_A24']
# subjects=['m2m_A24']
# subjects=['m2m_A2','m2m_A9']
# methods_folders={'Origin':'35','Baseline':'31_based_step1','PLED (Ours)':'39_based_step1'}
# subjects=['m2m_390645','m2m_432332','m2m_566454','m2m_644044','m2m_753150']
# subjects=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
# methods_folders={'Origin':'35_hcp','Baseline':'31_based_step1_hcp','PLED (Ours)':'39_based_step1_hcp'}
# methods_folders={'Origin':'35_optimization_angleres_1','Baseline':'31_based_gtlabel_optimization_angleres_1','PLED':'39_based_gtlabel_optimization_angleres_1'}
# methods_folders={'PLED_depth_32':'39_based_gtlabel_optimization','PLED_depth_64':'43_3_based_gtlabel_optimization_depth_64'}
# methods_folders={'PLED_scalar':'39_based_gtlabel_optimization','PLED_vector':'42_4_based_gtlabel_optimization_vector'}
methods_folders={'Origin':'45_optimization_vector','Baseline':'46_based_step1_optimization_vector','PLED':'42_4_based_step1_optimization_vector'}
# subjects=['m2m_TMS-015','m2m_TMS-040','m2m_TMS-076','m2m_TMS-103']
classnames={'gm':'GM','foreground':'Total'}
aspects={'best_id':'Top1','best_id_top3':'Top3','best_id_top5':'Top5'}
input_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health'
output_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/results_analysis'
# sub_foldername=r'box_plot_viridis_nolegend\total'
# sub_foldername=r'box_plot_viridis_nolegend_39_based_gt_step1\total'
# gt_foldername=r'gt'
gt_foldername=r'gt_optimization_vector'
sub_foldername=r'efield_vector_bar_plot_viridis_based_step1'
output_path=os.path.join(output_root_path,sub_foldername)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def append_value(total,item):
    if total is None:
        total=item.copy()
    else:
        total=np.concatenate((total,item))
    return total


for classname in classnames.keys():
    values=None
    methods=None
    topns=None
    for aspect in aspects.keys():
        for method in methods_folders.keys():


            # # 有两个方法时
            # if method=='PLED_vector':
            #     gt_foldername=r'gt_optimization_vector'
            # else:
            #     gt_foldername=r'gt_optimization'
                

            total_num = 0
            best_id_match = 0
            for subject in subjects:

                # if method=='Origin' and subject not in ['m2m_A2','m2m_A9']:
                #     continue

                gt_path=os.path.join(input_root_path,gt_foldername,'optimal_eval',subject.split('_')[-1]+'_'+classname+'_'+aspect+'.npy')
                gt_data=np.load(gt_path)
                predict_path=os.path.join(input_root_path,methods_folders[method],'optimal_eval',subject.split('_')[-1]+'_'+classname+'_best_id'+'.npy')
                predict_data=np.load(predict_path)
                total_num=total_num+len(gt_data)
                if aspect=='best_id':
                    best_id_match=best_id_match+np.sum(predict_data==gt_data)
                else:
                    predict_data_expand=predict_data[:,None].repeat(gt_data.shape[1],axis=1)
                    best_id_match=best_id_match+np.sum(np.any(predict_data_expand==gt_data,axis=1))
            if total_num==0:
                values = append_value(values, np.array([0]))
            else:
                values=append_value(values,np.array([best_id_match/total_num*100]))
            topns=append_value(topns,np.array([aspects[aspect]]))
            methods=append_value(methods,np.array([method]))
    data = pd.DataFrame({'Methods': methods, 'Top N': topns, 'Percent (%)': values})
    plt.figure()
    # ax=sns.barplot(data=data, x='Top N', y='Percent (%)', hue='Methods',  linewidth=1.5, palette='viridis')
    ax = sns.barplot(data=data, x='Top N', y='Percent (%)', hue='Methods', linewidth=2.5, palette='viridis')
    for patch in ax.patches:
        patch.set_edgecolor('black')
        # patch.set_linewidth(1.5)
        patch.set_linewidth(2.5)
    # ax.legend_.remove()
    output_class_path=os.path.join(output_path,classname)
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)
    plt.savefig(os.path.join(output_class_path,classname+'_optimization_best_id.png'),dpi=1200,bbox_inches='tight', pad_inches=0.1, format='png')
    # plt.savefig(os.path.join(output_class_path, subject.split('_')[-1] + classname + '_optimization_best_id.png'), dpi=1200, bbox_inches='tight', pad_inches=0.1, format='png')
