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
    efield_array, _=transform_orientation(efield,target_orientation)
    efield_array = np.squeeze(efield_array).astype(np.float32)
    efield_array = np.where(mask==0,0,efield_array)
    surface_file=nib.load(surface_path)
    vertices_temp = surface_file.darrays[0].data
    vertices = (np.hstack((vertices_temp, np.tile(1, (vertices_temp.shape[0], 1))))).dot(np.linalg.inv(new_affine).T)[:, :3]  # 将vertices的坐标从世界坐标系变换回到图像体素坐标系
    colors = np.zeros_like(efield_array, dtype=np.float32)
    colors[mask] = efield_array[mask]
    grid_x, grid_y, grid_z = np.arange(efield_array.shape[0]), np.arange(efield_array.shape[1]), np.arange(efield_array.shape[2])
    interpolating_function = RegularGridInterpolator((grid_x, grid_y, grid_z), colors, method="linear", bounds_error=False, fill_value=0)
    vertex_colors = interpolating_function(vertices).astype(np.float32)
    vertex_colors = smooth_colors(vertices, vertex_colors, sigma=sigma, num_neibors=num_neibors)
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
