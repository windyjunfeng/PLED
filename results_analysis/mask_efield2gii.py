# coding=utf-8
import numpy as np
import nibabel.gifti as gif
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
try:
    from simnibs.mesh_tools import mesh_io
except:
    pass

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
    
    # 检查vertices_world是否有NaN
    if np.any(np.isnan(vertices_world)) or np.any(np.isinf(vertices_world)):
        print(f"Warning: vertices_world contains NaN or Inf values in surface file: {surface_path}")
        return None
    
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


def mask_efield_vector_vis(mask_path, efield_vector_path, surface_path, mesh_path, output_gii_path_magnitude, 
                          output_gii_path_normal_component, mask_keep, target_orientation=('R','A','S'), 
                          sigma=10, num_neibors=30, clip_min=0.0, clip_max=0.999, 
                          scalp_normals_smoothing_steps=20):
    '''
    功能：生成灰质部分矢量电场的gii文件（模长和法向分量）
    :param mask_path: 分割文件的路径
    :param efield_vector_path: 个体空间下矢量电场nifti文件的路径（4D，最后一维是x/y/z分量）
    :param surface_path: 灰质gii文件路径
    :param mesh_path: mesh文件路径，用于计算法向
    :param output_gii_path_magnitude: 生成的矢量电场模长gii文件路径
    :param output_gii_path_normal_component: 生成的矢量电场法向分量模长gii文件路径
    :param mask_keep: 分割文件中灰质部分对应的索引
    :param target_orientation: 空间方向编码
    :param sigma: 高斯平滑的标准差
    :param num_neibors: 平滑过程中邻近顶点的数量
    :param clip_min: colormap下限对应电场强度值（升序排序）的比重
    :param clip_max: colormap上限对应电场强度值（升序排序）的比重
    :param scalp_normals_smoothing_steps: 法向平滑步数
    '''
    # 加载mask和矢量电场
    image = nib.load(mask_path)
    image_array, new_affine = transform_orientation(image, target_orientation)
    image_array = np.squeeze(image_array).astype(np.uint16)
    mask = np.isin(image_array, mask_keep)
    
    efield_vector = nib.load(efield_vector_path)
    efield_vector_array, efield_vector_affine = transform_orientation(efield_vector, target_orientation)
    efield_vector_array = np.squeeze(efield_vector_array).astype(np.float32)
    
    # 确保efield和mask使用相同的affine矩阵
    if not np.allclose(new_affine, efield_vector_affine, atol=1e-6):
        print("Warning: mask and efield_vector affine matrices differ, using mask affine")
    
    # 检查矢量电场的维度
    if efield_vector_array.ndim != 4:
        raise ValueError(f"Expected 4D vector field (x, y, z, components), got {efield_vector_array.ndim}D")
    if efield_vector_array.shape[3] != 3:
        raise ValueError(f"Expected 3 components (x, y, z), got {efield_vector_array.shape[3]}")
    
    # 提取x, y, z分量
    efield_x = efield_vector_array[:, :, :, 0]
    efield_y = efield_vector_array[:, :, :, 1]
    efield_z = efield_vector_array[:, :, :, 2]
    
    # 在mask外设置为0
    efield_x = np.where(mask == 0, 0, efield_x)
    efield_y = np.where(mask == 0, 0, efield_y)
    efield_z = np.where(mask == 0, 0, efield_z)
    
    # 计算矢量模长
    efield_magnitude = np.sqrt(efield_x**2 + efield_y**2 + efield_z**2)
    
    # 加载表面文件
    surface_file = nib.load(surface_path)
    vertices_world = surface_file.get_arrays_from_intent(1008)[0].data
    
    # 检查vertices_world是否有NaN
    if np.any(np.isnan(vertices_world)) or np.any(np.isinf(vertices_world)):
        print(f"Warning: vertices_world contains NaN or Inf values in surface file: {surface_path}")
        return None, None, None
    
    # 将vertices从世界坐标系转换到体素坐标系
    vertices_homogeneous = np.hstack((vertices_world, np.ones((vertices_world.shape[0], 1))))
    inv_affine = np.linalg.inv(new_affine)
    vertices_voxel = vertices_homogeneous.dot(inv_affine.T)[:, :3]
    
    # 加载mesh并计算灰质表面法向
    # 注意：mesh是全脑的，但局部gii文件的顶点坐标是世界坐标系下的
    # 所以可以通过在全脑mesh的灰质表面上找到最近点来计算法向
    mesh = mesh_io.read(mesh_path)
    msh_surf = mesh.crop_mesh(elm_type=2)  # 提取表面元素
    msh_gm = msh_surf.crop_mesh([2, 1002])  # 提取灰质表面（标签2和1002）
    
    # 计算每个顶点的法向（通过找到最近的表面点）
    # 注意：局部gii文件的顶点坐标是世界坐标系，全脑mesh也是世界坐标系
    # 所以可以直接在全脑mesh的灰质表面上找到最近点来计算法向
    vertex_normals = np.zeros_like(vertices_world)
    if len(msh_gm.elm.elm_number) > 0:
        # 获取所有三角形的法向（平滑后的）
        triangle_normals = msh_gm.triangle_normals(smooth=scalp_normals_smoothing_steps)
        
        # 获取所有三角形的重心，用于快速最近邻搜索
        triangle_centers = msh_gm.elements_baricenters()[:]
        tree = cKDTree(triangle_centers)
        
        # 对于每个局部表面的顶点，在全脑灰质表面上找到最近的三角形
        # 方法：先找到最近的三角形重心，然后使用该三角形的法向
        # 这是合理的，因为局部表面的顶点应该接近全脑mesh的灰质表面
        for i, vertex in enumerate(vertices_world):
            # 找到最近的三角形（基于重心）
            _, nearest_idx = tree.query(vertex, k=1)
            vertex_normals[i] = triangle_normals[nearest_idx]
        
        # 可选：验证法向的合理性（检查距离是否在合理范围内）
        # 如果局部表面顶点距离全脑mesh表面太远，可能需要警告
        distances_to_centers = np.linalg.norm(triangle_centers[tree.query(vertices_world, k=1)[1]] - vertices_world, axis=1)
        max_distance = float(np.percentile(distances_to_centers, 95))  # 95百分位数，转换为Python float以便JSON序列化
        if max_distance > 5.0:  # 如果距离超过5mm，给出警告
            print(f"Warning: Some local surface vertices are far from global mesh surface (max distance: {max_distance:.2f}mm)")
            print("This may indicate coordinate system mismatch or surface reconstruction issues")
    else:
        print("Warning: No gray matter surface elements found in mesh")
        max_distance = None  # 如果没有找到灰质表面，返回None
    
    # 插值矢量电场的x, y, z分量到表面顶点
    grid_x = np.arange(efield_x.shape[0], dtype=np.float32)
    grid_y = np.arange(efield_x.shape[1], dtype=np.float32)
    grid_z = np.arange(efield_x.shape[2], dtype=np.float32)
    
    # 插值x分量
    colors_x = np.zeros_like(efield_x, dtype=np.float32)
    colors_x[mask] = efield_x[mask]
    interpolating_function_x = RegularGridInterpolator((grid_x, grid_y, grid_z), colors_x, 
                                                       method="linear", bounds_error=False, fill_value=0)
    vertex_efield_x = interpolating_function_x(vertices_voxel).astype(np.float32)
    
    # 插值y分量
    colors_y = np.zeros_like(efield_y, dtype=np.float32)
    colors_y[mask] = efield_y[mask]
    interpolating_function_y = RegularGridInterpolator((grid_x, grid_y, grid_z), colors_y, 
                                                       method="linear", bounds_error=False, fill_value=0)
    vertex_efield_y = interpolating_function_y(vertices_voxel).astype(np.float32)
    
    # 插值z分量
    colors_z = np.zeros_like(efield_z, dtype=np.float32)
    colors_z[mask] = efield_z[mask]
    interpolating_function_z = RegularGridInterpolator((grid_x, grid_y, grid_z), colors_z, 
                                                       method="linear", bounds_error=False, fill_value=0)
    vertex_efield_z = interpolating_function_z(vertices_voxel).astype(np.float32)
    
    # 处理mask外的点（使用最近邻）
    valid_mask = (vertices_voxel[:, 0] >= 0) & (vertices_voxel[:, 0] < efield_x.shape[0]) & \
                 (vertices_voxel[:, 1] >= 0) & (vertices_voxel[:, 1] < efield_x.shape[1]) & \
                 (vertices_voxel[:, 2] >= 0) & (vertices_voxel[:, 2] < efield_x.shape[2])
    
    mask_coords = np.argwhere(mask)
    if len(mask_coords) > 0:
        mask_tree = cKDTree(mask_coords)
    else:
        mask_tree = None
    
    v_int_all = np.round(vertices_voxel).astype(int)
    v_int_all = np.clip(v_int_all, [0, 0, 0], 
                       [efield_x.shape[0]-1, efield_x.shape[1]-1, efield_x.shape[2]-1])
    mask_check = mask[v_int_all[:, 0], v_int_all[:, 1], v_int_all[:, 2]]
    invalid_indices = np.where(valid_mask & (~mask_check))[0]
    
    if len(invalid_indices) > 0 and mask_tree is not None:
        invalid_vertices = vertices_voxel[invalid_indices]
        k_neighbors = min(4, len(mask_coords))
        
        if k_neighbors == 1:
            distances, nearest_indices = mask_tree.query(invalid_vertices, k=1)
            if np.isscalar(nearest_indices):
                nearest_indices = np.array([nearest_indices])
            nearest_coords = mask_coords[nearest_indices]
            vertex_efield_x[invalid_indices] = efield_x[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]]
            vertex_efield_y[invalid_indices] = efield_y[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]]
            vertex_efield_z[invalid_indices] = efield_z[nearest_coords[:, 0], nearest_coords[:, 1], nearest_coords[:, 2]]
        else:
            distances, nearest_indices = mask_tree.query(invalid_vertices, k=k_neighbors)
            epsilon = 1e-6
            weights = 1.0 / (distances**2 + epsilon)
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            for i, idx in enumerate(invalid_indices):
                indices_row = nearest_indices[i]
                weights_row = weights[i]
                coords = mask_coords[indices_row]
                vertex_efield_x[idx] = np.sum(efield_x[coords[:, 0], coords[:, 1], coords[:, 2]] * weights_row)
                vertex_efield_y[idx] = np.sum(efield_y[coords[:, 0], coords[:, 1], coords[:, 2]] * weights_row)
                vertex_efield_z[idx] = np.sum(efield_z[coords[:, 0], coords[:, 1], coords[:, 2]] * weights_row)
    
    vertex_efield_x[~valid_mask] = 0
    vertex_efield_y[~valid_mask] = 0
    vertex_efield_z[~valid_mask] = 0
    
    # 计算顶点处的矢量模长
    vertex_magnitude = np.sqrt(vertex_efield_x**2 + vertex_efield_y**2 + vertex_efield_z**2)
    
    # 计算法向分量的模长
    # 法向分量 = 矢量 · 法向单位向量
    # 归一化法向
    normal_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    normal_norms[normal_norms == 0] = 1  # 避免除零
    vertex_normals_normalized = vertex_normals / normal_norms
    
    # 计算法向分量（点积）
    vertex_efield_vector = np.stack([vertex_efield_x, vertex_efield_y, vertex_efield_z], axis=1)  # (N_vertices, 3)
    normal_component = np.sum(vertex_efield_vector * vertex_normals_normalized, axis=1)  # (N_vertices,)
    vertex_normal_component_magnitude = np.abs(normal_component)  # 取绝对值
    
    # 平滑处理
    vertex_magnitude = smooth_colors(vertices_voxel, vertex_magnitude, sigma=sigma, num_neibors=num_neibors)
    vertex_normal_component_magnitude = smooth_colors(vertices_voxel, vertex_normal_component_magnitude, 
                                                      sigma=sigma, num_neibors=num_neibors)
    
    # 处理NaN值
    vertex_magnitude = np.where(np.isnan(vertex_magnitude), 0, vertex_magnitude)
    vertex_normal_component_magnitude = np.where(np.isnan(vertex_normal_component_magnitude), 0, 
                                                 vertex_normal_component_magnitude)
    
    # 裁剪值范围
    def clip_values(values, clip_min, clip_max):
        total_points = values.shape[0]
        ids = np.argsort(values)
        if clip_min == 0.0:
            v_min = values.min()
        else:
            v_min = values[ids[round(total_points * clip_min)]]
        if clip_max == 1.0:
            v_max = values.max()
        else:
            v_max = values[ids[round(total_points * clip_max)]]
        return np.clip(values, v_min, v_max)
    
    vertex_magnitude = clip_values(vertex_magnitude, clip_min, clip_max)
    vertex_normal_component_magnitude = clip_values(vertex_normal_component_magnitude, clip_min, clip_max)
    
    # 保存模长
    gii_file_magnitude = gif.GiftiImage()
    gii_file_magnitude.add_gifti_data_array(gif.GiftiDataArray(vertex_magnitude))
    nib.save(gii_file_magnitude, output_gii_path_magnitude)
    
    # 保存法向分量模长
    gii_file_normal = gif.GiftiImage()
    gii_file_normal.add_gifti_data_array(gif.GiftiDataArray(vertex_normal_component_magnitude))
    nib.save(gii_file_normal, output_gii_path_normal_component)
    
    return vertex_magnitude, vertex_normal_component_magnitude, max_distance
