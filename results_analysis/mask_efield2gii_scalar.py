# coding=utf-8
import numpy as np
import nibabel.gifti as gif
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


def smooth_colors(vertices, vertex_colors, sigma=10, num_neibors=30):
    '''
    功能：平滑顶点的颜色
    :param vertices: 顶点坐标
    :param vertex_colors: 顶点目前对应的颜色
    :param sigma: 高斯平滑的标准差
    :param num_neibors: 平滑过程中邻近顶点的数量
    :return: smoothed_colors：平滑后顶点的颜色
    '''
    tree = cKDTree(vertices)
    smoothed_colors = np.copy(vertex_colors)
    for i, vertex in enumerate(vertices):
        # 查找邻近的顶点
        distances, neighbors = tree.query(vertex, k=num_neibors)  # 可调整邻居数量k
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights /= weights.sum()
        smoothed_colors[i] = np.dot(weights, vertex_colors[neighbors])
    return smoothed_colors


def transform_orientation(image_nib, target_orientation=('R','A','S')):
    '''
    功能: 空间编码变换，对应的nifti数据和affine矩阵都要转换
    :param image_nib: nifti图像
    :param target_orientation: 空间方向编码
    :return:
    image_array: 转换后的nifti图像数据
    new_affien: 转换后的affine矩阵
    '''
    image_array = image_nib.get_fdata()
    current_orientation = nib.orientations.aff2axcodes(image_nib.affine)
    if current_orientation == target_orientation:
        new_affine = image_nib.affine
    else:
        ornt_transform = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(current_orientation),
                                                         nib.orientations.axcodes2ornt(target_orientation))
        new_affine = image_nib.affine @ nib.orientations.inv_ornt_aff(ornt_transform, image_nib.shape)
        image_array = nib.orientations.apply_orientation(image_array, ornt_transform)
    return image_array, new_affine


def mask_efield_vis(mask_path,efield_path,surface_path,output_gii_path,mask_keep,target_orientation=('R','A','S'),sigma=10, num_neibors=30, clip_min=0.0,clip_max=0.999):
    '''
    功能：生成灰质部分的电场gii文件
    :param mask_path: 分割文件的路径
    :param efield_path: 个体空间下电场nifti文件的路径
    :param surface_path: 灰质gii文件路径
    :param output_gii_path: 生成的灰质部分电场gii文件路径
    :param mask_keep: 分割文件中灰质部分对应的索引
    :param target_orientation: 空间方向编码
    :param sigma: 高斯平滑的标准差
    :param num_neibors: 平滑过程中邻近顶点的数量
    :param clip_min: colormap下限对应电场强度值（升序排序）的比重
    :param clip_max: colormap上限对应电场强度值（升序排序）的比重
    '''
    image = nib.load(mask_path)
    image_array, new_affine=transform_orientation(image,target_orientation)
    image_array=np.squeeze(image_array).astype(np.uint16)
    mask = np.isin(image_array,mask_keep)
    efield = nib.load(efield_path)
    efield_array, efield_affine=transform_orientation(efield,target_orientation)
    efield_array = np.squeeze(efield_array).astype(np.float32)
    efield_array = np.where(mask==0,0,efield_array)
    
    # 确保efield和mask使用相同的affine矩阵
    if not np.allclose(new_affine, efield_affine, atol=1e-6):
        print("Warning: mask and efield affine matrices differ, using mask affine")
    
    surface_file=nib.load(surface_path)
    # vertices_world = surface_file.darrays[0].data  # 世界坐标系下的顶点
    vertices_world = surface_file.get_arrays_from_intent(1008)[0].data
    
    # 将vertices从世界坐标系转换到体素坐标系，使用更精确的方法
    # 使用齐次坐标进行变换
    vertices_homogeneous = np.hstack((vertices_world, np.ones((vertices_world.shape[0], 1))))
    inv_affine = np.linalg.inv(new_affine)
    vertices_voxel = vertices_homogeneous.dot(inv_affine.T)[:, :3]  # 将vertices的坐标从世界坐标系变换回到图像体素坐标系
    
    # 确保插值网格使用浮点数，提高精度
    colors = np.zeros_like(efield_array, dtype=np.float32)
    colors[mask] = efield_array[mask]
    
    # 使用更精确的网格定义（从0到shape-1，而不是整数网格）
    grid_x = np.arange(efield_array.shape[0], dtype=np.float32)
    grid_y = np.arange(efield_array.shape[1], dtype=np.float32)
    grid_z = np.arange(efield_array.shape[2], dtype=np.float32)
    
    # 使用线性插值，但确保边界处理正确
    interpolating_function = RegularGridInterpolator((grid_x, grid_y, grid_z), colors, 
                                                     method="linear", bounds_error=False, fill_value=0)
    vertex_colors = interpolating_function(vertices_voxel).astype(np.float32)
    
    # 确保只在mask内部进行插值
    # 注意：在脑沟处，表面向内凹陷，插值可能会取到脑沟内部的体素值
    # 这里我们确保只使用mask内部的体素值，避免取到脑沟内部的非灰质体素
    # 检查顶点是否在有效范围内
    valid_mask = (vertices_voxel[:, 0] >= 0) & (vertices_voxel[:, 0] < efield_array.shape[0]) & \
                 (vertices_voxel[:, 1] >= 0) & (vertices_voxel[:, 1] < efield_array.shape[1]) & \
                 (vertices_voxel[:, 2] >= 0) & (vertices_voxel[:, 2] < efield_array.shape[2])
    
    # 对于mask外的点，使用最近邻方法检查：如果插值位置不在mask内，使用最近的mask内体素的值
    # 优化：预先构建mask内体素坐标的KDTree，用于快速最近邻搜索
    mask_coords = np.argwhere(mask)  # 获取所有mask内体素的坐标
    if len(mask_coords) > 0:
        mask_tree = cKDTree(mask_coords)  # 构建KDTree用于快速最近邻搜索
    else:
        mask_tree = None
    
    # 向量化处理：先检查所有顶点的最近邻体素是否在mask内
    v_int_all = np.round(vertices_voxel).astype(int)
    v_int_all = np.clip(v_int_all, [0, 0, 0], 
                       [efield_array.shape[0]-1, efield_array.shape[1]-1, efield_array.shape[2]-1])
    
    # 批量检查哪些顶点的最近邻体素不在mask内
    mask_check = mask[v_int_all[:, 0], v_int_all[:, 1], v_int_all[:, 2]]
    invalid_indices = np.where(valid_mask & (~mask_check))[0]  # 需要查找最近邻的顶点索引
    
    # 对于不在mask内的顶点，使用距离加权的多个最近邻插值（比单个最近邻更精确）
    if len(invalid_indices) > 0 and mask_tree is not None:
        invalid_vertices = vertices_voxel[invalid_indices]
        # 查询k个最近邻（k=4，使用距离加权插值，比单个最近邻更平滑精确）
        k_neighbors = min(4, len(mask_coords))  # 最多使用4个最近邻
        
        if k_neighbors == 1:
            # 单个最近邻的情况
            distances, nearest_indices = mask_tree.query(invalid_vertices, k=1)
            if np.isscalar(nearest_indices):
                nearest_indices = np.array([nearest_indices])
            if np.isscalar(distances):
                distances = np.array([distances])
            # 直接使用最近邻的值
            nearest_coords = mask_coords[nearest_indices]
            vertex_colors[invalid_indices] = efield_array[nearest_coords[:, 0], 
                                                         nearest_coords[:, 1], 
                                                         nearest_coords[:, 2]]
        else:
            # 多个最近邻，使用反距离加权插值（Inverse Distance Weighting, IDW）
            distances, nearest_indices = mask_tree.query(invalid_vertices, k=k_neighbors)
            
            # 使用反距离加权插值：权重 = 1 / (distance^2 + epsilon)
            # epsilon防止除零，并确保距离为0时权重为1
            epsilon = 1e-6
            weights = 1.0 / (distances**2 + epsilon)
            weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化权重
            
            # 批量计算加权平均
            interpolated_values = np.zeros(len(invalid_indices), dtype=np.float32)
            for i in range(len(invalid_indices)):
                indices_row = nearest_indices[i]
                weights_row = weights[i]
                coords = mask_coords[indices_row]
                values = efield_array[coords[:, 0], coords[:, 1], coords[:, 2]]
                # 加权平均
                interpolated_values[i] = np.sum(values * weights_row)
            
            # 批量更新vertex_colors
            vertex_colors[invalid_indices] = interpolated_values
    
    # 对于不在有效范围内的顶点，设置为0
    vertex_colors[~valid_mask] = 0
    
    vertex_colors = smooth_colors(vertices_voxel, vertex_colors, sigma=sigma, num_neibors=num_neibors)
    vertex_colors = np.where(np.isnan(vertex_colors), 0, vertex_colors)
    total_points = vertex_colors.shape[0]
    ids = np.argsort(vertex_colors)
    if clip_min == 0.0:
        v_min = vertex_colors.min()
    else:
        v_min = vertex_colors[ids[round(total_points * clip_min)]]
    if clip_max == 1.0:
        v_max = vertex_colors.max()
    else:
        v_max = vertex_colors[ids[round(total_points * clip_max)]]
    vertex_colors = np.clip(vertex_colors, v_min, v_max)
    gii_file = gif.GiftiImage()
    gii_file.add_gifti_data_array(gif.GiftiDataArray(vertex_colors))
    nib.save(gii_file,output_gii_path)
    return vertex_colors
