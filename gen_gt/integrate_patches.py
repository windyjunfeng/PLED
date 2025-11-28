import os
import json
import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm


def integrate_patches_with_affine(patches, patch_affines, original_affine, original_shape, method='average', label_list=None):
    """
    使用affine矩阵将局部影像块整合回原始MRI空间
    
    参数:
    - patches: 局部影像块列表
    - patch_affines: 每个patch对应的4×4 affine矩阵
    - original_affine: 原始MRI的affine矩阵
    - original_shape: 原始MRI的形状
    - method: 重叠区域合并方法
    """
    
    # 初始化输出体积和权重图
    integrated_volume = np.zeros(original_shape, dtype=np.float32)
    weight_volume = np.zeros(original_shape, dtype=np.float32)
    
    # 若使用投票法，预先为每个类别建立计数体积
    if method == 'vote':
        # 假设分割标签为非负整数类别，从patch中估计类别数
        # max_label = int(max(np.max(patch) for patch in patches)) if len(patches) > 0 else 0
        # num_classes = max_label + 1
        num_classes = len(label_list)
        label_to_index = {lbl: idx for idx, lbl in enumerate(label_list)}
        # votes 形状: (num_classes, *original_shape)，记录每体素各类别票数
        votes = np.zeros((num_classes,) + tuple(original_shape), dtype=np.int32)
    
    for i, (patch, patch_affine) in tqdm(enumerate(zip(patches, patch_affines)), total=len(patches), desc="整合patches"):
        # print(f"处理第 {i+1}/{len(patches)} 个patch...")
        
        # 计算从patch体素空间到原始体素空间的变换
        # world -> original_voxel = inv(original_affine)
        # patch_voxel -> world = patch_affine
        # 所以: patch_voxel -> original_voxel = inv(original_affine) @ patch_affine
        # transform_to_original = patch_affine @ np.linalg.inv(original_affine)
        transform_to_original = np.linalg.inv(original_affine)@patch_affine
        
        # 使用仿射变换将patch映射到原始空间
        transformed_patch = affine_transform_3d(
            patch, 
            transform_to_original, 
            output_shape=original_shape,
        )
        
        # 创建变换后的权重图（用于处理重叠）
        weight_patch = affine_transform_3d(
            np.ones_like(patch),
            transform_to_original,
            output_shape=original_shape,
        )
        
        if method == 'average':
            # 累加到结果中
            integrated_volume += transformed_patch
            weight_volume += weight_patch
        elif method == 'vote':
            # 仅对映射覆盖到的体素计票
            covered_mask = weight_patch > 0
            if np.any(covered_mask):
                # 对每个类别进行一次掩码累加，适合小类别数的分割问题
                tp_int = transformed_patch.astype(np.int64, copy=False)
                for cls_name in label_list:
                    cls_mask = covered_mask & (tp_int == cls_name)
                    if np.any(cls_mask):
                        votes[label_to_index[cls_name]][cls_mask] += 1
    
    # 处理重叠区域
    if method == 'average':
        non_zero_mask = weight_volume > 0
        integrated_volume[non_zero_mask] = integrated_volume[non_zero_mask] / weight_volume[non_zero_mask]
    elif method == 'vote':
        # 以多数票为最终结果
        # argmax 得到的是索引，需要映射回真实标签
        majority_indices = np.argmax(votes, axis=0)
        label_array = np.array(label_list, dtype='i2')
        majority_labels = label_array[majority_indices]
        total_votes = np.sum(votes, axis=0).astype('i2')
        integrated_volume = majority_labels.astype('i2')
        weight_volume = total_votes
    
    return integrated_volume, weight_volume

def affine_transform_3d(data, affine, output_shape):
    """
    对3D数据应用仿射变换 - 将patch数据变换到原始图像空间
    
    参数:
    - data: 输入3D数据（patch）
    - affine: 4×4仿射矩阵（从patch空间到原始图像空间）
    - output_shape: 输出形状（原始图像形状）
    """
    # 初始化输出数组
    transformed = np.zeros(output_shape, dtype=data.dtype)
    
    # 获取patch的形状
    patch_shape = data.shape
    
    # 生成patch中所有体素的坐标
    i_coords, j_coords, k_coords = np.mgrid[0:patch_shape[0], 0:patch_shape[1], 0:patch_shape[2]]
    
    # 将所有patch体素坐标转换为齐次坐标
    patch_coords = np.stack([
        i_coords.ravel(), 
        j_coords.ravel(), 
        k_coords.ravel(), 
        np.ones(i_coords.ravel().shape)
    ]).T  # shape: (N, 4)
    
    # 批量变换到原始图像空间
    # original_coords = (affine @ patch_coords.T).T  # shape: (N, 4)
    original_coords = patch_coords.dot(affine.T)
    x_coords = original_coords[:, 0]
    y_coords = original_coords[:, 1] 
    z_coords = original_coords[:, 2]
    
    # 四舍五入到最近的整数坐标（用于索引）
    x_indices = np.round(x_coords).astype(int)
    y_indices = np.round(y_coords).astype(int)
    z_indices = np.round(z_coords).astype(int)
    
    # 创建掩码，筛选出在原始图像范围内的坐标
    valid_mask = ((x_indices >= 0) & (x_indices < output_shape[0]) &
                  (y_indices >= 0) & (y_indices < output_shape[1]) &
                  (z_indices >= 0) & (z_indices < output_shape[2]))
    
    if np.any(valid_mask):
        # 获取有效坐标和对应的patch值
        valid_x_indices = x_indices[valid_mask]
        valid_y_indices = y_indices[valid_mask]
        valid_z_indices = z_indices[valid_mask]
        valid_values = data.ravel()[valid_mask]
        
        # 将patch值赋值到原始图像中
        transformed[valid_x_indices, valid_y_indices, valid_z_indices] = valid_values
    
    return transformed


# def integrate_with_nibabel(patches, patch_affines, original_affine, original_shape):
#     """
#     使用nibabel工具进行整合
#     """
#     integrated_volume = np.zeros(original_shape, dtype=np.float32)
#     weight_volume = np.zeros(original_shape, dtype=np.float32)
    
#     for patch, patch_affine in tqdm(zip(patches, patch_affines), total=len(patches), desc="使用nibabel整合patches"):
#         # 创建patch的nibabel图像
#         patch_img = nib.Nifti1Image(patch, patch_affine)
        
#         # 创建目标空间的nibabel图像
#         target_img = nib.Nifti1Image(np.zeros(original_shape), original_affine)
        
#         # 使用nibabel的resampling方法
#         from nibabel.processing import resample_from_to
        
#         try:
#             # 将patch重采样到目标空间
#             resampled_data = resample_from_to(patch_img, target_img, order=1)[0].get_fdata()
            
#             # 创建权重图
#             weight_patch = resample_from_to(
#                 nib.Nifti1Image(np.ones_like(patch), patch_affine), 
#                 target_img, order=0
#             )[0].get_fdata()
            
#             integrated_volume += resampled_data
#             weight_volume += weight_patch
            
#         except Exception as e:
#             print(f"重采样失败: {e}")
#             continue
    
#     # 平均值合并
#     non_zero_mask = weight_volume > 0
#     integrated_volume[non_zero_mask] = integrated_volume[non_zero_mask] / weight_volume[non_zero_mask]
    
#     return integrated_volume, weight_volume


if __name__ == "__main__":
    # patch_dir = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion/m2m_TMS-015/local_sampling_scalp_imgs_new/local_image'
    patch_dir =r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion/m2m_TMS-015/local_sampling_scalp_labels_new/local_label'
    affine_json_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion/m2m_TMS-015/local_sampling_scalp_imgs_new/affine_matrices_local_imgs.json'
    original_img_path =r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion/m2m_TMS-015/T1_origin_resample_noneck.nii.gz'
    # output_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion/m2m_TMS-015/T1_integrated.nii.gz'
    output_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion/m2m_TMS-015/final_tissues_integrated.nii.gz'
    # method = 'average'  # 'average':平均值, 'max':最大值
    method = 'vote'  # 针对分割标签的整合
    # label_list = [1,2,3,4,5,9,11,12,13,14,19]
    label_list = [0,1,2,3,4,5,9]
    # order = 1  # 0:最近邻, 1:线性, 3:三次

    with open(affine_json_path, 'r') as f:
        affine_json = json.load(f)
    patch_files = os.listdir(patch_dir)
    patches = []
    patch_affines = []
    for patch_file in patch_files:
        patch_name = patch_file.split('.')[0]
        if patch_name.split('_')[-1] == '0' and int(patch_name.split('_')[-2])%4==0 :  # 每个采样点只取一个方向的patch,并且采样点的索引是4的倍数
            patch_path = os.path.join(patch_dir, patch_file)
            patch = np.load(patch_path)
            patch_affine = affine_json[patch_name]
            patches.append(patch)
            patch_affines.append(patch_affine)
    # 加载原始MRI获取affine和shape
    original_img = nib.load(original_img_path)
    original_affine = original_img.affine
    original_shape = original_img.shape


    # 方法1: 自定义实现
    integrated_volume, weight_map = integrate_patches_with_affine(
        patches, patch_affines, original_affine, original_shape, method=method, label_list=label_list
    )

    # # 方法2: 使用nibabel（推荐，更稳定）
    # integrated_volume, weight_map = integrate_with_nibabel(
    #     patches, patch_affines, original_affine, original_shape
    # )

    # 保存结果
    result_img = nib.Nifti1Image(integrated_volume, original_affine)
    nib.save(result_img, output_path)