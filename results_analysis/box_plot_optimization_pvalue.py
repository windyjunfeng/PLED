# coding=utf-8

import os
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size':15})


subjects=['m2m_A2','m2m_A3','m2m_A8','m2m_A9','m2m_A14','m2m_A17','m2m_A18','m2m_A20','m2m_A24']
# methods_folders={'Origin':'35','Baseline':'31_based_step1','PLED (Ours)':'39_based_step1'}
# methods_folders={'Origin':'35_add_lesion','Baseline':'31_based_step1_add_lesion','PLED (Ours)':'39_based_step1_add_lesion'}
# methods_folders={'Origin':'35_optimization_angleres_1','Baseline':'31_based_gtlabel_optimization_angleres_1','PLED':'39_based_gtlabel_optimization_angleres_1'}
# methods_folders={'PLED_depth_32':'39_based_gtlabel_optimization','PLED_depth_64':'43_3_based_gtlabel_optimization_depth_64'}
# methods_folders={'PLED_scalar':'39_based_gtlabel_optimization','PLED_vector':'42_4_based_gtlabel_optimization_vector'}
methods_folders={'Origin':'45_optimization_vector','Baseline':'46_based_gtlabel_optimization_vector','PLED':'42_4_based_gtlabel_optimization_vector'}
# subjects=['m2m_390645','m2m_432332','m2m_566454','m2m_644044','m2m_753150']
# subjects=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
# subjects=['m2m_A2','m2m_A9']
# subjects=['m2m_TMS-015','m2m_TMS-040','m2m_TMS-076','m2m_TMS-103']
# subjects=['m2m_753150']
# methods_folders={'Origin':'35_cog','Baseline':'31_based_step1_cog','PLED':'39_based_step1_cog'}
classnames={'gm':'GM','foreground':'Total'}
# aspects={'pos':'Distance of optimal coil positions (mm)','angle':'MAE of optimal coil orientation (°)','efield':'MRE of optimal E-field magnitude (%)'}
aspects={'angle':'MAE of optimal coil orientation (°)','efield':'MRE of optimal E-field magnitude (%)'}
input_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health'
output_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/results_analysis'
# sub_foldername=r'box_plot_viridis_nolegend\total'
# sub_foldername=r'box_plot_viridis_nolegend_39_based_gt_step1\total'
# gt_foldername=r'gt'
gt_foldername=r'gt_optimization_vector'
# sub_foldername=r'efield_scalar_vector_box_plot_viridis_pvalue'
sub_foldername=r'efield_vector_box_plot_viridis_pvalue'
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
    categories=None
    methods=None
    for aspect in aspects.keys():
        for method in methods_folders.keys():


            # # 有两个方法时
            # if method=='PLED_vector':
            #     gt_foldername=r'gt_optimization_vector'
            # else:
            #     gt_foldername=r'gt_optimization'


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
                    # eval=np.abs(predict_data-gt_data)*15 # MAE

                mask=np.isfinite(eval)
                values_filtered=eval[mask]  # 过滤掉为nan或inf的元素
                method_item=np.array([method]*len(values_filtered))
                values=append_value(values,values_filtered)
                methods=append_value(methods,method_item)
        data = pd.DataFrame({'Methods': methods, aspects[aspect]: values})
        plt.figure()
        ax = sns.boxplot(data=data, x='Methods', y=aspects[aspect], hue='Methods', linewidth=2.5, width=0.5,
                         showfliers=False, palette='viridis')  # 不显示离群点
        # ax=sns.boxplot(data=data, x='Methods', y=aspects[aspect], hue='Methods', linewidth=1.5, width=0.5, showfliers=False, palette='viridis')  # 不显示离群点
        # ax.legend_.remove()
        # import pdb
        # pdb.set_trace()
        p_values = []


        # 有三个方法时
        pairwise_results = pg.pairwise_tests(dv=aspects[aspect], between='Methods', data=data, parametric=True, padjust='bonf')  # T检验，Bonferroni correction，参数化检验，pairwise_results.columns可以查看结果字典中的key有哪些
        p_values.append([pairwise_results['p-corr'][2], pairwise_results['p-corr'][1]])
        

        # # 有两个方法时
        # pairwise_results = pg.pairwise_tests(dv=aspects[aspect], between='Methods', data=data, parametric=True)# 只有两个方法时不进行Bonferroni correction
        # p_values.append([pairwise_results['p-unc'][0]])
        

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin


        # 有三个方法时
        for j in range(2):  
            if j == 0:
                width_left = 1
                width_right = 1
                # height_line = 0.05
                # height_star = 0.035
                height_line = 0.045
                height_star = 0.02
                p_val = p_values[0][j]
                
            else:
                width_left = 0
                width_right = 1
                height_line = -0.01
                # height_star = -0.025  # 由于*并不是贴着底线写的，因此需要向下平移
                height_star = -0.03
                p_val = p_values[0][j]
                    # 绘制横线
            plt.plot([1 - width_left, 1 + width_right],
                        # [ymax + height_line * y_range, ymax + height_line * y_range], color='black', lw=1.5)
                        [ymax + height_line * y_range, ymax + height_line * y_range], color='black', lw=2.5)
            # 添加星号表示显著性
            if p_val < 0.05 and p_val >= 0.01:
                plt.text(1 + (width_right - width_left) / 2, ymax + height_star * y_range, f"*", ha='center',
                            # va='bottom', fontsize=12)
                            va='bottom', fontsize=16)
            elif p_val < 0.01 and p_val >= 0.001:
                plt.text(1 + (width_right - width_left) / 2, ymax + height_star * y_range, f"**", ha='center',
                            # va='bottom', fontsize=12)
                            va='bottom', fontsize=16)
            elif p_val < 0.001:
                plt.text(1 + (width_right - width_left) / 2, ymax + height_star * y_range, f"***", ha='center',
                            # va='bottom', fontsize=12)
                            va='bottom', fontsize=16)
            else:
                # plt.text(1 + (width_right - width_left) / 2, ymax + (height_star+0.015) * y_range, f"n.s.", ha='center',
                            # va='bottom', fontsize=10)
                plt.text(1 + (width_right - width_left) / 2, ymax + (height_star + 0.02) * y_range, f"n.s.",ha='center',
                            va='bottom', fontsize=14)



        # # 有两个方法时
        # width_left = 1
        # width_right = 0
        # height_line = 0.045
        # height_star = 0.02
        # p_val = p_values[0][0]
        # # 绘制横线
        # plt.plot([1 - width_left, 1 + width_right],
        #             # [ymax + height_line * y_range, ymax + height_line * y_range], color='black', lw=1.5)
        #             [ymax + height_line * y_range, ymax + height_line * y_range], color='black', lw=2.5)
        # # 添加星号表示显著性
        # if p_val < 0.05 and p_val >= 0.01:
        #     plt.text(1 + (width_right - width_left) / 2, ymax + height_star * y_range, f"*", ha='center',
        #                 # va='bottom', fontsize=12)
        #                 va='bottom', fontsize=16)
        # elif p_val < 0.01 and p_val >= 0.001:
        #     plt.text(1 + (width_right - width_left) / 2, ymax + height_star * y_range, f"**", ha='center',
        #                 # va='bottom', fontsize=12)
        #                 va='bottom', fontsize=16)
        # elif p_val < 0.001:
        #     plt.text(1 + (width_right - width_left) / 2, ymax + height_star * y_range, f"***", ha='center',
        #                 # va='bottom', fontsize=12)
        #                 va='bottom', fontsize=16)
        # else:
        #     # plt.text(1 + (width_right - width_left) / 2, ymax + (height_star+0.015) * y_range, f"n.s.", ha='center',
        #                 # va='bottom', fontsize=10)
        #     plt.text(1 + (width_right - width_left) / 2, ymax + (height_star + 0.02) * y_range, f"n.s.",ha='center',
        #                 va='bottom', fontsize=14)


        output_class_path=os.path.join(output_path,classname)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        # plt.savefig(os.path.join(output_class_path,aspect+'_optimization.png'),dpi=1200,bbox_inches='tight', pad_inches=0.1, format='png')
        plt.savefig(os.path.join(output_class_path, aspect + '_optimization.png'), dpi=1200, format='png')
