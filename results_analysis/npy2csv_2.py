# coding=utf-8
import os
import numpy as np
import pandas as pd


methods_folders={'Origin':'35_cog','Baseline':'31_based_step1_cog','PLED':'39_based_step1_cog'}
# methods_folders={'Origin':'35_s','Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}
metric_representation={'pearson_corr':'PCC'}
# metric_representation={'mae':'MAE','mre':'MRE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mre_loss_edge':'ARE','mae_loss_percentile95':'MAE_95%tile'}
input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\final_significance_analysis\total_generalization_test\cog'

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
    group_id=1
    group_id_total=None
    value_total=None
    for method in methods_folders.keys():
        data=np.load(os.path.join(input_root_path,methods_folders[method],'total',filename))
        mask = np.isfinite(data)
        data_filtered = data[mask]  # 过滤掉为nan或inf的元素
        group_id_item=np.array([group_id]*len(data_filtered))
        group_id_total=append_value(group_id_total,group_id_item)
        value_total=append_value(value_total,data_filtered)
        group_id = group_id + 1
    df = pd.DataFrame({'group_id':group_id_total,metric_representation[key]:value_total})
    df.to_csv(os.path.join(output_path, filename.split('.')[0] + '.csv'),index=False)
