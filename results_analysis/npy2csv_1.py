# coding=utf-8
import os
import numpy as np
import pandas as pd


methods_folders={'Origin':'35_s','Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}
metric_representation={'mae':'MAE','mre':'MRE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mre_loss_edge':'ARE','mae_loss_percentile95':'MAE_95%tile'}
input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\final_significance_analysis'
# input_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\39_final_new_based_step1\total'
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\39_final_new_based_step1\total_csv'
# input_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\35_s\total'
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\35_s\total_csv'
# input_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\31_based_step1\total'
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\31_based_step1\total_csv'
if not os.path.exists(output_path):
    os.makedirs(output_path)


def append_value(total,item):
    if total is None:
        total=item.copy()
    else:
        total=np.concatenate((total,item))
    return total


npy_filenames=os.listdir(input_root_path)
for key in metric_representation.keys():
    filename=key+'_total.npy'
    data_filtered={}
    for method in methods_folders.keys():
        data=np.load(os.path.join(input_root_path,methods_folders[method],'total',filename))
        mask = np.isfinite(data)
        data_filtered[method] = data[mask]  # 过滤掉为nan或inf的元素
    max_length=max(len(value) for value in data_filtered.values())  # 生成pandas格式的数据需要字典中元素的长度相等，因此需要对较短者填充nan
    for k,v in data_filtered.items():
        padding_length=max_length-len(v)
        if padding_length>0:
            data_filtered[k]=np.pad(v,(0,padding_length),mode='constant',constant_values=np.nan)
    df = pd.DataFrame(data_filtered)
    df.to_csv(os.path.join(output_path, filename.split('.')[0] + '.csv'),index=False)
