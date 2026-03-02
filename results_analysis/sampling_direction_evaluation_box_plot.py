# coding=utf-8
import os
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


metric_representation={'mae':'MAE','mre':'MRE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mre_loss_edge':'ARE'}
# classnames={'total':'total', 'class_2':'GM'}
classnames={'total':'total'}
methods_folders={'Origin':'35_s','Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}
# methods_folders={'Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}  # 'Origin':'35_s'只有m2m_A2和m2m_A9
test_subjects=['m2m_A2','m2m_A9','m2m_A14','m2m_A20','m2m_A3','m2m_A8','m2m_A17','m2m_A18','m2m_A24']
# methods_folders={'Origin':'35_s'}
# test_subjects=['m2m_A2','m2m_A9']
num_directions=12
input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
sampling_info_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\test_subjects\val_l_sampling_info'
output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis'
sub_foldername=r'sampling_direction_evaluation_box_plot'
output_path=os.path.join(output_root_path,sub_foldername)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def append_value(total,item):
    if total is None:
        total=item.copy()
    else:
        total=np.concatenate((total,item))
    return total


sorted_names_total=None
for test_subject in test_subjects:
    sampling_info_path=os.path.join(sampling_info_root_path,test_subject,'affine_matrices_local_labels.json')
    with open(sampling_info_path,'r',encoding='utf-8') as file:
        sampling_info=json.load(file)
    sorted_names=np.array(sorted(list(sampling_info.keys())))  # 按ASCII码排序，使得和evalute时val_dataloader加载的顺序保持一致
    sorted_names_total=append_value(sorted_names_total,sorted_names)


# import pdb
# pdb.set_trace()
for key in metric_representation.keys():
    for classname in classnames.keys():
        indicator_stop_falg = False
        methods = None
        names_groups = None
        values_filtered_total = None
        output_class_path = os.path.join(output_path,classname)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        for method in methods_folders.keys():
            if method == 'Origin':  #Origin方法只有两个测试数据
                sorted_names_total_temp=sorted_names_total[:101268]
            else:
                sorted_names_total_temp=sorted_names_total
            path=os.path.join(os.path.join(input_root_path,methods_folders[method]),classname,key+'_'+classname+'.npy')
            values_origin=np.load(path)
            if values_origin.shape[0]!=sorted_names_total_temp.shape[0]:
                print(key,'_',method,'_',classname,': part of the data indicators are lost!')  # 由于该类指标有方法不行，因此就无法整体比较了，因此也无需统计其他方法该类指标的数据了,直接break
                indicator_stop_falg=True
                break
            mask=np.isfinite(values_origin)
            values_filtered=values_origin[mask]  # 过滤掉为nan或inf的元素
            names_filtered=sorted_names_total_temp[mask]
            names_group=np.array([name.split('_')[-1] for name in names_filtered])
            methods=append_value(methods,np.array([method]*len(names_group)))
            names_groups=append_value(names_groups,names_group)
            values_filtered_total=append_value(values_filtered_total,values_filtered)
        # import pdb
        # pdb.set_trace()
        if indicator_stop_falg:
            continue
        df=pd.DataFrame({metric_representation[key]:values_filtered_total,'Direction (°)':names_groups.astype(int),'Methods':methods})
        df_sorted_1=df.sort_values(by='Direction (°)', ascending=True)
        desired_order=['Origin','Baseline','PLED']
        df_sorted_1['Methods']=pd.Categorical(df_sorted_1['Methods'],categories=desired_order,ordered=True)  # 确保显示顺序按照指定的顺序
        df_sorted_final=df_sorted_1.sort_values(by='Methods')
        plt.figure()
        ax=sns.boxplot(data=df_sorted_final, x='Direction (°)', y=metric_representation[key], hue='Methods', linewidth=1.5, showfliers=False, palette='viridis')  # 不显示离群点
        ax.legend_.remove()
        plt.savefig(os.path.join(output_class_path, metric_representation[key] + '_category.png'), dpi=1200, bbox_inches='tight', pad_inches=0.1, format='png')