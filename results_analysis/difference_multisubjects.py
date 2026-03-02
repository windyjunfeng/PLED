# coding=utf-8
import os
import json
from scipy.stats import levene
import numpy as np


metric_representation={'mae':'MAE','mre':'MRE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mre_loss_edge':'ARE'}
classnames={'total':'total'}
methods_folders={'Origin':'35_s','Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}
test_subjects=['m2m_A2','m2m_A9','m2m_A14','m2m_A20','m2m_A3','m2m_A8','m2m_A17','m2m_A18','m2m_A24']
input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
sampling_info_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\test_subjects\val_l_sampling_info'
output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis'
sub_foldername=r'difference_multisubjects'
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


for key in metric_representation.keys():
    for classname in classnames.keys():
        indicator_stop_falg = False
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

        if indicator_stop_falg:
            continue
