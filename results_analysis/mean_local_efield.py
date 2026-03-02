# coding=utf-8
import os
import numpy as np
from tqdm import tqdm

def extract_fields(filename):
    # 分割文件名，获取多个字段
    prefilename,_=os.path.splitext(filename)
    parts = prefilename.split('_')
    # 假设我们想按第二个和第三个字段排序
    return int(parts[1]), int(parts[2])  # 返回一个元组


subjects=['m2m_A2','m2m_A3','m2m_A8','m2m_A9','m2m_A14','m2m_A17','m2m_A18','m2m_A20','m2m_A24']
# subjects=['m2m_A2','m2m_A9']
# subjects=['m2m_390645','m2m_432332','m2m_566454','m2m_644044','m2m_753150']
# subjects=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
# subjects=['m2m_TMS-015','m2m_TMS-040','m2m_TMS-076','m2m_TMS-103']
# root_efield_path=r'/root/autodl-tmp/data/local_efield'
# root_seg_path=r'/root/autodl-tmp/data/seg'
# root_output_path=r'/root/autodl-tmp/data/output_31_based_step1'
# root_efield_path=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/inference/35_cog'
# root_seg_path=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/32_cog'
# root_efield_path=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield'
# root_seg_path=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
# root_output_path=r'/root/autodl-tmp/tms_e-field_scalp_data/35_cog'
root_efield_path=r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
root_seg_path=r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
# root_output_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/gt_optimization_angleres_1/mean_efield'
root_output_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/35_optimization_angleres_1/mean_efield'
if not os.path.exists(root_output_path):
    os.makedirs(root_output_path)
for subject in subjects:
    foreground_mean_efield_list=[]
    gm_mean_efield_list=[]
    # subject_efield_path=os.path.join(root_efield_path,subject,'local_efield')
    # subject_efield_path = os.path.join(root_efield_path, subject, 'local_efield', subject)
    # subject_efield_path = os.path.join(root_efield_path, subject)
    # subject_efield_path = os.path.join(root_efield_path, subject,'mesh2nii_index_new_correct_interpolate_angleres_1/efield_calculation/local_efield')
    subject_efield_path = os.path.join(root_efield_path, subject,'inference/local_efield_35')
    subject_seg_path=os.path.join(root_seg_path,subject,'local_sampling_scalp_labels_new_1010_center_angleres_1/local_label')
    # subject_seg_path=os.path.join(root_seg_path,subject)
    filenames=os.listdir(subject_seg_path)
    # filenames=os.listdir(subject_efield_path)
    filenames_sorted=sorted(filenames,key=extract_fields)  # 根据采样点的id进行排序，每个采样点根据角度的大小进行排序，这样才能保证后续使用时直接对应coil_positions.npy的顺序
    # import pdb
    # pdb.set_trace()
    with tqdm(total=len(filenames_sorted)) as pbar:
        for filename in filenames_sorted:
            efield=np.load(os.path.join(subject_efield_path,filename))
            if len(efield.shape)==4 and efield.shape[-1]==3:  # 如果是矢量电场，则转换为幅值
                efield=np.linalg.norm(efield, axis=-1)
            seg=np.load(os.path.join(subject_seg_path,filename))
            foreground_flag=(seg!=0)
            # gm_flag=(seg==2)
            gm_flag=np.logical_or(seg==2, seg==12) # 包括灰质和灰质部分的病灶
            if np.all(foreground_flag==False):
                foreground_mean_efield=np.nan
            else:
                foreground_efield=efield[foreground_flag]
                foreground_mean_efield=np.mean(foreground_efield)
            if np.all(gm_flag==False):
                gm_mean_efield=np.nan
            else:
                gm_efield=efield[gm_flag]
                gm_mean_efield=np.mean(gm_efield)
            # foreground_mean_efield=np.mean(foreground_efield)
            # gm_mean_efield=np.mean(gm_efield)
            foreground_mean_efield_list.append(foreground_mean_efield)
            gm_mean_efield_list.append(gm_mean_efield)
            pbar.update(1)
    foreground_mean_efield_array=np.array(foreground_mean_efield_list).astype('float16')
    gm_mean_efield_array=np.array(gm_mean_efield_list).astype('float16')
    np.save(os.path.join(root_output_path,subject.split('_')[-1]+'_foreground_mean_efield.npy'),foreground_mean_efield_array)
    np.save(os.path.join(root_output_path, subject.split('_')[-1] + '_gm_mean_efield.npy'),gm_mean_efield_array)
    # pbar.update(1)
