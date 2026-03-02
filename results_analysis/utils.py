# coding=utf-8
import nibabel as nib
import nibabel.gifti as gif
import numpy as np
from skimage import measure
from scipy.ndimage import label
from skimage.measure import regionprops
import trimesh


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


def remove_small_objects(image, min_size=1000):
    """
    功能: 移除小的连通组件。
    :param image: 输入的二值图像。
    :param min_size: 要保留的最小连通组件大小。
    :return: 处理后的图像
    """
    # 连通组件标记
    labeled_image, num_labels = label(image)

    # 获取连通组件属性
    regions = regionprops(labeled_image)

    # 创建一个空白的输出图像
    result_image = np.zeros_like(image, dtype=np.uint8)

    # 遍历所有连通组件并保留大于min_size的组件
    for region in regions:
        if region.area >= min_size:
            # 将符合条件的连通组件添加到结果图像中
            result_image[labeled_image == region.label] = 1

    return result_image


def laplacian_smooth(vertices, faces, iterations=10, lambda_factor=0.5):
    '''
    功能：拉普拉斯平滑表面
    :param vertices: 模型的顶点坐标
    :param faces: 模型的面（顶点索引）
    :param iterations: 拉普拉斯平滑迭代次数
    :param lambda_factor: 拉普拉斯平滑因子
    :return: 平滑后的模型的顶点坐标
    '''
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    for _ in range(iterations):
        neighbors = mesh.vertex_neighbors
        new_vertices = vertices.copy()
        for i in range(len(vertices)):
            neighbor_verts = vertices[neighbors[i]]
            new_vertices[i] = vertices[i] + lambda_factor * (np.mean(neighbor_verts, axis=0) - vertices[i])
        vertices = new_vertices
    return vertices


def close_bottom_quarter(data,x_start,x_end,y_start,y_end,fill_value):
    '''
    功能：一方面是为了使得后续生成的头皮模型封闭，另一方面是为了使得填洞操作origin=(0,0,1)时能够较为正常
    :param data: 输入数据
    :param x_start: x轴起始位置
    :param x_end: x轴终止位置
    :param y_start: y轴起始位置
    :param y_end: y轴终止位置
    :param fill_value: 填充值
    :return: data: 封底后的数据
    '''
    if x_start<x_end:
        x_step=1
    else:
        x_step=-1
    if y_start<y_end:
        y_step=1
    else:
        y_step=-1
    for x_i in range(x_start,x_end,x_step):
        for y_i in range(y_start,y_end,y_step):
            if data[x_i,y_i,-1]==fill_value:
                data[x_i:x_end:x_step,y_i:y_end:y_step,:]=fill_value
                break
    return data


def mask2gii(mask,affine,output_surface_path,iterations=10,lambda_factor=0.5):
    '''
    功能：重建mask的表面生成gii文件
    :param mask: 待表面重建的numpy array
    :param affine: 待表面重建的numpy array到世界坐标系下的仿射变换矩阵
    :param output_surface_path: 输出gii文件路径
    :param iterations: 拉普拉斯平滑迭代次数
    :param lambda_factor: 拉普拉斯平滑因子
    '''
    verts, faces, _, _ = measure.marching_cubes(mask, level=0)
    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int32))
    mesh_smoothed = trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, lamb=lambda_factor)  # 平滑
    verts_temp = mesh_smoothed.vertices
    faces_temp = mesh_smoothed.faces
    verts_smoothed, indices_temp = np.unique(verts_temp, axis=0, return_inverse=True)  # 去除重复的顶点
    verts_smoothed = np.hstack((verts_smoothed, np.tile(1, (verts_smoothed.shape[0], 1))))
    verts_smoothed_affine = verts_smoothed.dot(affine.T)[:, :3]  # 顶点坐标变换到个体空间下的世界坐标系中
    new_faces_temp = indices_temp[faces_temp]  # 更新三角形的顶点索引
    faces_smoothed_temp, _ = np.unique(new_faces_temp, axis=0, return_inverse=True)  # 去除重复的三角形
    faces_smoothed = np.array([row for row in faces_smoothed_temp if len(row) == len(set(row))])  # 去除异常的三角形（顶点有重合）
    gii_file = gif.GiftiImage()
    vertices_data = gif.GiftiDataArray(verts_smoothed_affine.astype(np.float32), intent='NIFTI_INTENT_POINTSET')
    faces_data = gif.GiftiDataArray(faces_smoothed.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE')
    gii_file.add_gifti_data_array(vertices_data)
    gii_file.add_gifti_data_array(faces_data)
    nib.save(gii_file, output_surface_path)