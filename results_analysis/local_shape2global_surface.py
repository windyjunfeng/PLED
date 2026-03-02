# coding=utf-8
"""
功能：将局部的shape.gii文件映射到全局的surf.gii上，生成全局的shape.gii文件
超出局部范围的部分：通过过渡因子平滑过渡到0，距离边界越远的越趋于0
"""
import numpy as np
import nibabel.gifti as gif
import nibabel as nib
from scipy.spatial import cKDTree
from utils import transform_orientation


def smooth_colors(vertices, vertex_colors, sigma=10, num_neibors=30):
    '''
    功能：平滑顶点的颜色（批量查询+向量化，提升速度）
    :param vertices: 顶点坐标
    :param vertex_colors: 顶点目前对应的颜色
    :param sigma: 高斯平滑的标准差
    :param num_neibors: 平滑过程中邻近顶点的数量
    :return: smoothed_colors：平滑后顶点的颜色
    '''
    tree = cKDTree(vertices)
    # 批量查询所有顶点的邻居，避免逐点循环
    distances, neighbors = tree.query(vertices, k=num_neibors)
    # 向量化高斯权重：(n_vertices, k)
    weights = np.exp(-distances**2 / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    # 用邻居索引取颜色：(n_vertices, k)，再加权求和
    neighbor_colors = vertex_colors[neighbors]
    smoothed_colors = (weights * neighbor_colors).sum(axis=1)
    return smoothed_colors.astype(vertex_colors.dtype)


def map_local_shape_to_global_surface(global_mask_nii_path, local_shape_gii_path, 
                                     local_surf_gii_path, global_surf_gii_path,
                                     output_global_shape_gii_path, mask_keep,
                                     target_orientation=('R', 'A', 'S'),
                                     decay_scale=5.0, smooth_sigma=10.0, smooth_neighbors=30):
    '''
    功能：将局部的shape.gii文件映射到全局的surf.gii上，生成全局的shape.gii文件
    :param global_mask_nii_path: 全局分割结果.nii.gz文件路径
    :param local_shape_gii_path: 局部shape.gii文件路径（包含电场值）
    :param local_surf_gii_path: 局部surf.gii文件路径（用于确定局部空间范围）
    :param global_surf_gii_path: 全局surf.gii文件路径
    :param output_global_shape_gii_path: 输出的全局shape.gii文件路径
    :param mask_keep: 分割文件中灰质部分对应的索引
    :param target_orientation: 空间方向编码
    :param decay_scale: 衰减尺度（单位：mm），控制超出局部区域后值衰减到0的速度
                        使用指数衰减函数：exp(-distance / decay_scale)
                        值越大，衰减越慢（值在更远距离仍保持较大，过渡更平滑）
                        值越小，衰减越快（值更快地接近0，过渡更陡峭）
                        例如：decay_scale=5.0时，距离边界5mm处的值约为原来的1/e≈0.368
    :param smooth_sigma: 全局颜色平滑的高斯标准差（单位：mm），用于使过渡更自然
                         值越大，平滑范围越大；值越小，平滑范围越小
    :param smooth_neighbors: 全局颜色平滑时使用的邻近顶点数量
    :return: global_vertex_colors: 全局顶点的颜色值
    '''
    # 1. 读取全局mask，确定全局空间范围
    global_mask_nii = nib.load(global_mask_nii_path)
    global_mask_array, global_affine = transform_orientation(global_mask_nii, target_orientation)
    global_mask_array = np.squeeze(global_mask_array).astype(np.uint16)
    global_mask = np.isin(global_mask_array, mask_keep)
    
    # 2. 读取局部surf.gii，获取局部顶点坐标（世界坐标系）
    local_surf_gii = nib.load(local_surf_gii_path)
    local_vertices_world = local_surf_gii.get_arrays_from_intent(1008)[0].data  # 世界坐标系下的顶点
    
    # 3. 读取局部shape.gii，获取局部顶点的颜色值
    local_shape_gii = nib.load(local_shape_gii_path)
    local_vertex_colors = local_shape_gii.darrays[0].data  # 局部顶点的颜色值
    
    if len(local_vertex_colors.shape) > 1:
        local_vertex_colors = local_vertex_colors.flatten()
    
    # 确保局部顶点和颜色值数量一致
    if local_vertices_world.shape[0] != local_vertex_colors.shape[0]:
        raise ValueError(f"局部surf.gii的顶点数({local_vertices_world.shape[0]})与局部shape.gii的颜色值数量({local_vertex_colors.shape[0]})不匹配")
    
    # 4. 读取全局surf.gii，获取全局顶点坐标（世界坐标系）
    global_surf_gii = nib.load(global_surf_gii_path)
    global_vertices_world = global_surf_gii.get_arrays_from_intent(1008)[0].data  # 世界坐标系下的顶点
    
    # 5. 计算局部顶点的边界框（用于确定局部空间范围）
    local_bbox_min = local_vertices_world.min(axis=0)
    local_bbox_max = local_vertices_world.max(axis=0)
    
    # 6. 构建局部顶点的KDTree，用于快速最近邻搜索
    local_vertices_tree = cKDTree(local_vertices_world)
    
    # 7. 对于全局surf.gii的每个顶点，确定其颜色值
    global_vertex_colors = np.zeros(global_vertices_world.shape[0], dtype=np.float32)
    
    # 判断每个全局顶点是否在局部边界框内
    in_local_bbox = np.all((global_vertices_world >= local_bbox_min) & 
                          (global_vertices_world <= local_bbox_max), axis=1)
    
    # 8. 处理在局部边界框内的顶点：直接使用局部shape.gii的值（通过最近邻插值）
    if np.any(in_local_bbox):
        local_bbox_vertices = global_vertices_world[in_local_bbox]
        # 使用k个最近邻进行插值（k=3，使用距离加权）
        k_neighbors = min(3, len(local_vertices_world))
        distances, nearest_indices = local_vertices_tree.query(local_bbox_vertices, k=k_neighbors)
        
        if k_neighbors == 1:
            # 单个最近邻
            if np.isscalar(nearest_indices):
                nearest_indices = np.array([nearest_indices])
            if np.isscalar(distances):
                distances = np.array([distances])
            global_vertex_colors[in_local_bbox] = local_vertex_colors[nearest_indices]
        else:
            # 多个最近邻，使用反距离加权插值（向量化）
            epsilon = 1e-6
            weights = 1.0 / (distances**2 + epsilon)
            weights = weights / weights.sum(axis=1, keepdims=True)  # 归一化权重
            values = local_vertex_colors[nearest_indices]  # (n_bbox, k)
            global_vertex_colors[in_local_bbox] = (values * weights).sum(axis=1)
    
    # 9. 处理在局部边界框外的顶点：根据距离边界越远，值越趋于0（通过过渡因子平滑）
    out_local_bbox = ~in_local_bbox
    if np.any(out_local_bbox):
        outside_vertices = global_vertices_world[out_local_bbox]
        
        # 计算每个顶点到局部顶点集合的最近距离
        distances_to_local, _ = local_vertices_tree.query(outside_vertices, k=1)
        
        # 计算到局部边界框表面的距离（向量化）
        dists_per_dim = np.zeros((len(outside_vertices), 3))
        for dim in range(3):
            v = outside_vertices[:, dim]
            dists_per_dim[:, dim] = np.where(
                v < local_bbox_min[dim], local_bbox_min[dim] - v,
                np.where(v > local_bbox_max[dim], v - local_bbox_max[dim],
                         np.minimum(v - local_bbox_min[dim], local_bbox_max[dim] - v))
            )
        distances_to_bbox_surface = np.linalg.norm(dists_per_dim, axis=1)
        
        # 使用边界框表面距离和局部顶点距离的较小值，更准确地反映边界距离
        boundary_distances = np.minimum(distances_to_local, distances_to_bbox_surface)
        
        # 获取边界外顶点的插值颜色值
        k_neighbors = min(3, len(local_vertices_world))
        distances, nearest_indices = local_vertices_tree.query(outside_vertices, k=k_neighbors)
        
        if k_neighbors == 1:
            if np.isscalar(nearest_indices):
                nearest_indices = np.array([nearest_indices])
            outside_colors = local_vertex_colors[nearest_indices]
        else:
            epsilon = 1e-6
            weights = 1.0 / (distances**2 + epsilon)
            weights = weights / weights.sum(axis=1, keepdims=True)
            values = local_vertex_colors[nearest_indices]  # (n_outside, k)
            outside_colors = (values * weights).sum(axis=1)
        
        # 应用衰减：距离越远，颜色值越接近0
        # 使用指数衰减函数：exp(-distance / decay_scale)
        # decay_scale越大，衰减越慢（过渡更平滑）；decay_scale越小，衰减越快（过渡更陡峭）
        transition_weights = np.exp(-boundary_distances / decay_scale)
        global_vertex_colors[out_local_bbox] = outside_colors * transition_weights
    
    # 10. 处理NaN值
    global_vertex_colors = np.where(np.isnan(global_vertex_colors), 0, global_vertex_colors)
    
    # 11. 对全局顶点颜色进行平滑处理，使过渡更自然
    global_vertex_colors = smooth_colors(global_vertices_world, global_vertex_colors, 
                                         sigma=smooth_sigma, num_neibors=smooth_neighbors)
    global_vertex_colors = np.where(np.isnan(global_vertex_colors), 0, global_vertex_colors)
    
    # 12. 保存全局shape.gii文件
    gii_file = gif.GiftiImage()
    gii_file.add_gifti_data_array(gif.GiftiDataArray(global_vertex_colors))
    nib.save(gii_file, output_global_shape_gii_path)
    
    return global_vertex_colors


if __name__ == "__main__":
    # 示例用法
    global_mask_nii_path = r'/data/disk_2/zhoujunfeng/data/temp/m2m_sub002/final_tissues_integrated_infer_dl_gm.nii.gz'
    local_shape_gii_path = r'/data/disk_2/zhoujunfeng/data/temp/m2m_sub002/visualization/sub002_0_-30/sub002_0_-30_efield.shape.gii'
    local_surf_gii_path = r'/data/disk_2/zhoujunfeng/data/temp/m2m_sub002/visualization/sub002_0_-30/sub002_0_-30_gm.surf.gii'
    global_surf_gii_path = r'/data/disk_2/zhoujunfeng/data/temp/m2m_sub002/cortex.surf.gii'
    output_global_shape_gii_path = r'/data/disk_2/zhoujunfeng/data/temp/m2m_sub002/visualization/sub002_0_-30/sub002_0_-30_efield_global.shape.gii'
    mask_keep = [2]
    
    global_vertex_colors = map_local_shape_to_global_surface(
        global_mask_nii_path=global_mask_nii_path,
        local_shape_gii_path=local_shape_gii_path,
        local_surf_gii_path=local_surf_gii_path,
        global_surf_gii_path=global_surf_gii_path,
        output_global_shape_gii_path=output_global_shape_gii_path,
        mask_keep=mask_keep,
        target_orientation=('R', 'A', 'S'),
        decay_scale=20.0,  # 衰减尺度（mm），控制超出局部区域后值衰减到0的速度
                          # 值越大，衰减越慢（过渡更平滑）；值越小，衰减越快（过渡更陡峭）
        smooth_sigma=10.0,  # 全局颜色平滑的高斯标准差（mm），用于使过渡更自然
        smooth_neighbors=10  # 全局颜色平滑时使用的邻近顶点数量
    )
    
    print(f"成功生成全局shape.gii文件，包含{len(global_vertex_colors)}个顶点的颜色值")
    print(f"颜色值范围: [{global_vertex_colors.min():.4f}, {global_vertex_colors.max():.4f}]")

