# coding=utf-8
import os
import numpy as np
import json
import pandas as pd


metric_representation={'mae':'MAE','mre':'MRE','f_norm':'F-Norm Loss','psnr':'PSNR','pearson_corr':'PCC','mre_loss_edge':'ARE'}
classnames={'total':'total', 'class_2':'GM'}
# methods_folders={'Baseline':'31_based_step1','PLED':'39_final_new_based_step1'}  # 'Origin':'35_s'只有m2m_A2和m2m_A9
# test_subjects=['m2m_A2','m2m_A9','m2m_A14','m2m_A20','m2m_A3','m2m_A8','m2m_A17','m2m_A18','m2m_A24']
methods_folders={'Origin':'35_s'}
test_subjects=['m2m_A2','m2m_A9']
num_directions=12
input_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation'
sampling_info_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\test_subjects\val_l_sampling_info'
output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis'
sub_foldername=r'sampling_points_evaluation_geo'
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
sampling_pos_total=None
for test_subject in test_subjects:
    sampling_info_path=os.path.join(sampling_info_root_path,test_subject,'affine_matrices_local_labels.json')
    sampling_pos_path=os.path.join(sampling_info_root_path,test_subject,'coil_positions.npy')
    with open(sampling_info_path,'r',encoding='utf-8') as file:
        sampling_info=json.load(file)
    sorted_names=np.array(sorted(list(sampling_info.keys())))  # 按ASCII码排序，使得和evalute时val_dataloader加载的顺序保持一致
    sorted_names_total=append_value(sorted_names_total,sorted_names)
    sampling_pos=np.load(sampling_pos_path)
    sampling_pos_total=append_value(sampling_pos_total,sampling_pos)

# import pdb
# pdb.set_trace()
for key in metric_representation.keys():
    for method in methods_folders.keys():
        for classname in classnames.keys():
            path=os.path.join(os.path.join(input_root_path,methods_folders[method]),classname,key+'_'+classname+'.npy')
            values_origin=np.load(path)
            if values_origin.shape[0]!=sorted_names_total.shape[0]:
                print(key,'_',method,'_',classname,': part of the data indicators are lost!')
                continue
            mask=np.isfinite(values_origin)
            values_filtered=values_origin[mask]  # 过滤掉为nan或inf的元素
            names_filtered=sorted_names_total[mask]
            names_group=np.array([name.split('_')[:2] for name in names_filtered])
            group_1=names_group[:,0]
            group_2=names_group[:,1]
            df=pd.DataFrame({'value':values_filtered,'group_1':group_1,'group_2':group_2})
            grouped_means=df.groupby(['group_1','group_2'],sort=False)['value'].mean().reset_index()  # 用于将group_1和group_2作为普通的列,并且添加一个总的顺序索引，更直观,sort=False防止改变之前的顺序
            final_grouped_means=grouped_means.groupby('group_1',sort=False)
            for group_2_key,group_value in final_grouped_means:
                geo_root_path=os.path.join(output_path,group_2_key,key,method)
                if not os.path.exists(geo_root_path):
                    os.makedirs(geo_root_path)
                geo_path=os.path.join(geo_root_path,group_2_key+'_'+key+'_'+method+'_'+classname+'_sampling_points.geo')
                # import pdb
                # pdb.set_trace()
                with open(geo_path, 'w') as f:
                    f.write("View 'grids' {\n")
                    for index,row in group_value.iterrows():
                        f.write("SP(" + ", ".join([str(coord) for coord in sampling_pos_total[index]]) + "){"+str(row['value'])+"};\n")
                    f.write("};\n")