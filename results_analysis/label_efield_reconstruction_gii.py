# coding=utf-8
import os
import trimesh
import json
import numpy as np
import nibabel.gifti as gif
import nibabel as nib
# from skimage import measure
from utils import *
# from mask_efield2gii_scalar import mask_efield_vis
from mask_efield2gii import mask_efield_vis, mask_efield_vector_vis


def label_reconstruction(input_path, output_surface_path, mask_keep, target_orientation=('R', 'A', 'S'), iterations=10, lambda_factor=0.5):
    '''
    功能：mask重建成gii文件
    :param input_path: 输入mask文件路径
    :param output_surface_path: 输出gii文件路径
    :param mask_keep: 要保留的mask的标签值
    :param target_orientation: 空间方向编码
    :param iterations: 拉普拉斯平滑迭代次数
    :param lambda_factor: 拉普拉斯平滑因子
    '''
    image = nib.load(input_path)
    image_array, new_affine = transform_orientation(image, target_orientation)
    image_array=np.squeeze(image_array).astype(np.uint16)
    mask = np.isin(image_array,mask_keep)
    image_array[~mask]=0
    image_array[mask]=1
    mask2gii(image_array, new_affine, output_surface_path, iterations, lambda_factor)


if __name__=="__main__":
    # input_root_path = r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\test_subjects\val_l_sampling_info'
    # output_root_path = r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\visualization'
    # test_subject=['m2m_A2']
    # subject_index=0
    # # sampling_list=['A2_311_45','A2_311_-45']
    # # sampling_list = ['A2_1781_45', 'A2_1781_-45']
    # sampling_list = ['A2_1488_45', 'A2_1488_-45']
    # sampling_index=0
    # mask_keep = [2]
    # methods=['GT','PLED','Origin','Baseline']
    # with open(os.path.join(input_root_path,test_subject[subject_index],'affine_matrices_local_labels.json'), 'r') as f1:
    #     coil_matrices = json.load(f1)
    # for sampling_index in range(len(sampling_list)):
    #     label_data=np.load(os.path.join(input_root_path,test_subject[subject_index],'label',sampling_list[sampling_index]+'.npy'))
    #     label_data=np.squeeze(label_data)
    #     nifti_image = nib.Nifti1Image(label_data.astype(np.float32), coil_matrices[sampling_list[sampling_index]])
    #     output_sub_path = os.path.join(output_root_path,test_subject[subject_index],sampling_list[sampling_index])
    #     if not os.path.exists(output_sub_path):
    #         os.makedirs(output_sub_path)
    #     nii_label_path=os.path.join(output_sub_path,sampling_list[sampling_index]+'_label.nii.gz')
    #     nifti_image.to_filename(nii_label_path)
    #     output_surface_path=os.path.join(output_sub_path,sampling_list[sampling_index]+'_gm.surf.gii')
    #     label_reconstruction(nii_label_path, output_surface_path, mask_keep)    
    #     vertex_colors_list=[]
    #     for method in methods:
    #         efield_data=np.load(os.path.join(input_root_path,test_subject[subject_index],'efield',method,sampling_list[sampling_index]+'.npy'))
    #         efield_data=np.squeeze(efield_data)
    #         nifti_efield = nib.Nifti1Image(efield_data.astype(np.float32), coil_matrices[sampling_list[sampling_index]])
    #         nii_efield_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield_'+method+'.nii.gz')
    #         nifti_efield.to_filename(nii_efield_path)
    #         output_efield_gii_path=os.path.join(output_sub_path,sampling_list[sampling_index]+'_efield_'+method+'.shape.gii')
    #         vertex_colors=mask_efield_vis(nii_label_path, nii_efield_path, output_surface_path, output_efield_gii_path, mask_keep)
    #         vertex_colors_list.append(vertex_colors)
    #         if len(vertex_colors_list)>1:
    #             # vertex_colors_mre=np.nan_to_num(np.abs((vertex_colors_list[-1]-vertex_colors_list[0])/vertex_colors_list[0])*(vertex_colors_list[0]!=0),nan=100.0)
    #             vertex_colors_mre = np.abs(vertex_colors_list[-1] - vertex_colors_list[0])  # 分母直接除以局部影像块可能得最大值，这里取1
    #             gii_file_mre = gif.GiftiImage()
    #             gii_file_mre.add_gifti_data_array(gif.GiftiDataArray(vertex_colors_mre))
    #             nib.save(gii_file_mre, output_efield_gii_path.split('.shape.gii')[0]+'_mre.shape.gii')


    input_root_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    output_root_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    coil_matrices_relpath = r'local_sampling_scalp_labels_new_1010_center/affine_matrices_local_labels.json'
    seg_relpath = r'local_sampling_scalp_labels_new_1010_center/local_label'
    efield_relpath = r'mesh2nii_index_new_correct_interpolate_vector/efield_calculation/local_efield_voxel_coord'
    output_relpath = r'visualization/local_sampling_scalp_labels_new_1010_center_efield_vector'
    # efield_relpath = r'mesh2nii_index_new_correct_interpolate_angleres_1/efield_calculation/local_efield'
    # output_relpath = r'visualization/local_sampling_scalp_labels_new_1010_center_efield_scalar'
    test_subject_txt_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/val_l.txt'
    with open(test_subject_txt_path, 'r') as f0:
        test_subject_lines = f0.readlines()
        test_subject = [line.rstrip('\n') for line in test_subject_lines]
    # test_subject=['m2m_A2']
    # subject_index=0
    for subject_index in range(len(test_subject)):
        # sampling_list = ['A2_0_-30', 'A2_10_0', 'A2_0_30']
        # sampling_index=0
        sampling_list_npy = os.listdir(os.path.join(input_root_path,test_subject[subject_index],seg_relpath))
        sampling_list = [sampling_npy.split('.npy')[0] for sampling_npy in sampling_list_npy if os.path.exists(os.path.join(input_root_path,test_subject[subject_index],efield_relpath,sampling_npy))]
        mask_keep = [2]  # 2是灰质标签
        with open(os.path.join(input_root_path,test_subject[subject_index],coil_matrices_relpath), 'r') as f1:
            coil_matrices = json.load(f1)
        
        # 用于收集每个采样点的max_distance信息
        max_distance_info = {}
        # 用于收集有问题的采样点（vertices_world包含NaN）
        problematic_samplings = []
        
        for sampling_index in range(len(sampling_list)):
            label_data=np.load(os.path.join(input_root_path,test_subject[subject_index],seg_relpath,sampling_list[sampling_index]+'.npy'))
            label_data=np.squeeze(label_data)
            nifti_image = nib.Nifti1Image(label_data.astype(np.float32), coil_matrices[sampling_list[sampling_index]])
            output_sub_path = os.path.join(output_root_path,test_subject[subject_index],output_relpath,sampling_list[sampling_index])
            if not os.path.exists(output_sub_path):
                os.makedirs(output_sub_path)
            nii_label_path=os.path.join(output_sub_path,sampling_list[sampling_index]+'_label.nii.gz')
            nifti_image.to_filename(nii_label_path)
            output_surface_path=os.path.join(output_sub_path,sampling_list[sampling_index]+'_gm.surf.gii')
            label_reconstruction(nii_label_path, output_surface_path, mask_keep)    
            efield_data=np.load(os.path.join(input_root_path,test_subject[subject_index],efield_relpath,sampling_list[sampling_index]+'.npy'))
            efield_data=np.squeeze(efield_data)
            
            # 判断是标量电场还是矢量电场（矢量电场多一维，是x/y/z分量）
            if efield_data.ndim == 4 and efield_data.shape[3] == 3:
                # 矢量电场
                # 需要找到mesh文件路径
                subject_id = test_subject[subject_index].split('m2m_')[-1]
                mesh_path = os.path.join(input_root_path, test_subject[subject_index], subject_id + '.msh')
                
                if os.path.exists(mesh_path):
                    # 保存为4D nifti（x, y, z, components）
                    nifti_efield_vector = nib.Nifti1Image(efield_data.astype(np.float32), coil_matrices[sampling_list[sampling_index]])
                    nii_efield_vector_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield_vector.nii.gz')
                    nifti_efield_vector.to_filename(nii_efield_vector_path)
                    
                    # 输出路径
                    output_efield_magnitude_gii_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield_vector_magnitude.shape.gii')
                    output_efield_normal_gii_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield_vector_normal_component.shape.gii')
                    
                    # 调用矢量电场可视化函数
                    result = mask_efield_vector_vis(
                        nii_label_path, nii_efield_vector_path, output_surface_path, mesh_path,
                        output_efield_magnitude_gii_path, output_efield_normal_gii_path, mask_keep
                    )
                    # 检查是否有NaN问题（返回值为(None, None, None)表示有问题）
                    if result is None or (isinstance(result, tuple) and len(result) == 3 and result[0] is None):
                        problematic_samplings.append(sampling_list[sampling_index])
                        print(f"Skipping {sampling_list[sampling_index]} due to NaN in vertices_world")
                        continue
                    vertex_magnitude, vertex_normal_component, max_distance = result
                    # 保存max_distance信息
                    max_distance_info[sampling_list[sampling_index]] = max_distance
                    # print(f"Vector field visualization completed: magnitude and normal component saved")
                else:
                    # print(f"Error: Mesh file not found, cannot visualize vector field normal component")
                    # print(f"Falling back to magnitude-only visualization")
                    # 降级为只计算模长
                    efield_magnitude = np.sqrt(np.sum(efield_data**2, axis=3))
                    nifti_efield = nib.Nifti1Image(efield_magnitude.astype(np.float32), coil_matrices[sampling_list[sampling_index]])
                    nii_efield_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield_magnitude.nii.gz')
                    nifti_efield.to_filename(nii_efield_path)
                    output_efield_gii_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield_magnitude.shape.gii')
                    vertex_colors = mask_efield_vis(nii_label_path, nii_efield_path, output_surface_path, output_efield_gii_path, mask_keep)
                    # 检查是否有NaN问题
                    if vertex_colors is None:
                        problematic_samplings.append(sampling_list[sampling_index])
                        print(f"Skipping {sampling_list[sampling_index]} due to NaN in vertices_world")
                        continue
            else:
                # 标量电场
                nifti_efield = nib.Nifti1Image(efield_data.astype(np.float32), coil_matrices[sampling_list[sampling_index]])
                nii_efield_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield.nii.gz')
                nifti_efield.to_filename(nii_efield_path)
                output_efield_gii_path = os.path.join(output_sub_path, sampling_list[sampling_index] + '_efield.shape.gii')
                vertex_colors = mask_efield_vis(nii_label_path, nii_efield_path, output_surface_path, output_efield_gii_path, mask_keep)
                # 检查是否有NaN问题
                if vertex_colors is None:
                    problematic_samplings.append(sampling_list[sampling_index])
                    print(f"Skipping {sampling_list[sampling_index]} due to NaN in vertices_world")
                    continue
        
        # 保存每个subject的max_distance信息到json文件
        if max_distance_info:  # 只有当有矢量电场数据时才保存
            subject_id = test_subject[subject_index].split('m2m_')[-1]
            json_output_path = os.path.join(output_root_path, test_subject[subject_index], output_relpath, f'{subject_id}_95th_max_distance_info.json')
            json_output_dir = os.path.dirname(json_output_path)
            if not os.path.exists(json_output_dir):
                os.makedirs(json_output_dir)
            with open(json_output_path, 'w') as f_json:
                json.dump(max_distance_info, f_json, indent=4)
            print(f"Max distance info saved to: {json_output_path}")
        
        # 保存有问题的采样点到npy文件
        if problematic_samplings:
            subject_id = test_subject[subject_index].split('m2m_')[-1]
            problematic_output_path = os.path.join(output_root_path, test_subject[subject_index], output_relpath, f'{subject_id}_problematic_samplings.npy')
            problematic_output_dir = os.path.dirname(problematic_output_path)
            if not os.path.exists(problematic_output_dir):
                os.makedirs(problematic_output_dir)
            np.save(problematic_output_path, np.array(problematic_samplings))
            print(f"Problematic samplings saved to: {problematic_output_path} (count: {len(problematic_samplings)})")