import os
import json
import simnibs
import numpy as np
import nibabel as nib
from tqdm import tqdm
from simnibs.mesh_tools import mesh_io


def tangential_plane(mesh, pos, distance, radius, resolution_pos=1, scalp_normals_smoothing_steps=20):
    msh_surf = mesh.crop_mesh(elm_type=2)
    msh_skin = msh_surf.crop_mesh([5, 1005])
    target_skin = msh_skin.find_closest_element(pos)
    elm_center = msh_skin.elements_baricenters()[:]
    elm_mask_roi = np.linalg.norm(elm_center - target_skin, axis=1) < 1.2 * radius  # 找到皮肤靶点方圆1.2*r范围内的element
    elm_center_zeromean = (elm_center[elm_mask_roi] - np.mean(elm_center[elm_mask_roi], axis=0))
    msh_roi = msh_skin.crop_mesh(elements=msh_skin.elm.elm_number[elm_mask_roi])
    u, s, vh = np.linalg.svd(elm_center_zeromean)  # vh是3*3矩阵，s是3个奇异值
    vh = vh.transpose()
    coords_plane = np.array(
        np.meshgrid(
            np.linspace(-radius, radius, int(2 * radius / resolution_pos + 1)),
            np.linspace(-radius, radius, int(2 * radius / resolution_pos + 1)),
        )
    ).T.reshape(-1, 2)
    coords_plane = coords_plane[np.linalg.norm(coords_plane, axis=1) <= radius]
    center_id=np.argmin(np.linalg.norm(coords_plane, axis=1))
    coords_plane = np.dot(coords_plane, vh[:, :2].transpose()) + target_skin
    normals_roi = msh_roi.triangle_normals(smooth=scalp_normals_smoothing_steps)
    q1 = coords_plane + 1e2 * resolution_pos * vh[:, 2]
    q2 = coords_plane - 1e2 * resolution_pos * vh[:, 2]
    idx, pos = msh_roi.intersect_segment(q1, q2)
    intersections = idx[:, 0] == center_id
    c=coords_plane[center_id,:]
    intersect_pos = pos[intersections]
    intersect_triangles = idx[intersections, 1]
    dist = np.linalg.norm(c[None, :] - intersect_pos, axis=1)
    closest = np.argmin(dist)
    coords_normals=normals_roi[intersect_triangles[closest]]
    coords_mapped=intersect_pos[closest]
    coords_mapped += distance * coords_normals
    return coords_mapped, coords_normals


def main():
    distance = 0
    radius = 28  # 求线圈法向量时所需的局部邻域半径
    resolution_angle = 15
    start_angle = -90  # 角度正负的定义是按照线圈手柄与中轴线的夹角，顺时针偏离中轴线角度为负，逆时针偏离中轴线角度为正
    end_angle = 90
    local_block_x = 80  # 局部采样块x轴方向上的边长
    local_block_y = 80  # 局部采样块y轴方向上的边长
    local_block_z = 32  # 局部采样块z轴方向上的边长
    root_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field'
    subjects=os.listdir(root_path)
    for subject in tqdm(subjects):
        subject_path=os.path.join(root_path,subject)
        if os.path.isdir(subject_path):
            subject_id=subject.split('_')[-1]
            mesh_path = os.path.join(subject_path,subject_id+'.msh')
            img_path = os.path.join(subject_path,'T1_origin_resample_noneck.nii.gz')  # 切割过脖子后的T1影像
            # img_path = os.path.join(subject_path,'final_tissues.nii.gz')
            local_sampling_path = os.path.join(subject_path,'local_sampling_scalp_imgs_new')
            # local_sampling_path = os.path.join(subject_path, 'local_sampling_scalp_labels_new')
            npy_path = os.path.join(local_sampling_path,'coil_positions.npy')
            local_img_save_path=os.path.join(local_sampling_path,'local_image')
            # local_img_save_path = os.path.join(local_sampling_path, 'local_label')
            json_affine_matrix_path=os.path.join(local_sampling_path,'affine_matrices_local_imgs.json')
            # json_affine_matrix_path = os.path.join(local_sampling_path, 'affine_matrices_local_labels.json')
            if not os.path.exists(local_img_save_path):
                os.makedirs(local_img_save_path)
            mesh = mesh_io.read(mesh_path)
            img=nib.load(img_path)
            img_buffer=np.round(img.get_fdata()).astype(np.uint16)
            if img_buffer.ndim==4:
                img_buffer=np.squeeze(img_buffer,axis=-1)
            y_size, z_size, x_size = img_buffer.shape
            img_affine = img.affine
            img_affine_inverse = np.linalg.inv(img_affine)
            x_internal, y_internal, z_internal = np.mgrid[-int(local_block_x / 2):int(local_block_x / 2), -int(local_block_y / 2):int(local_block_y / 2), 0:-local_block_z:-1]  # (3,y_size,z_size,x_size)  #直接从头皮上采样
            internal_points = np.vstack((x_internal.ravel(),y_internal.ravel(),z_internal.ravel(),np.tile(1,x_internal.ravel().shape[0]))).T  # ravel()函数将多维数组变成一维数组,为方便坐标变换后边补1
            sampling_array=np.load(npy_path)
            with open(json_affine_matrix_path, 'w') as f:
                affine_matrices={}
                for i in range(sampling_array.shape[0]):
                    pos=sampling_array[i]
                    coords_mapped, normal_z = tangential_plane(mesh, pos, distance, radius,resolution_pos=1,scalp_normals_smoothing_steps=20)
                    for theta_d in list(range(start_angle,end_angle,resolution_angle)):
                        theta=np.deg2rad(theta_d)
                        if normal_z[2] == 0:
                            normal_y = np.array([0, 0, 1])
                        else:
                            normal_y = np.array([0, 1, -normal_z[1] / normal_z[2]])
                        normal_y=normal_y/np.linalg.norm(normal_y)  # L2正则化
                        normal_x=np.cross(normal_y,normal_z)
                        rotation_matrix_origin=np.array([normal_x,normal_y,normal_z]).transpose()  # 将线圈坐标系下的x轴,y轴和z轴的单位向量转换为列向量进行拼接,这里垂直于头皮向外为线圈坐标系的z轴正方向,因为要跟图像空间的z轴正方向保持一致
                        Rz = np.array((
                            (np.cos(theta), -np.sin(theta), 0),
                            (np.sin(theta), np.cos(theta), 0),
                            (0, 0, 1),))
                        rotation_matrix = rotation_matrix_origin.dot(Rz)  # 由于是在rotation_matrix_origin基础上变换的，因此要右乘新的变换，此时这个偏航角是在偏航角为0的切面上定义的；若在世界坐标系下变换要左乘新的变换
                        translation_matrix=pos[:,None]
                        coil_matrix=np.concatenate((np.concatenate((rotation_matrix,translation_matrix),axis=1),[[0,0,0,1]]),axis=0)  # 该矩阵与线圈坐标系下的坐标相乘即得到世界坐标系中的坐标
                        image_points=(internal_points.dot(coil_matrix.T)).dot(img_affine_inverse.T)[:,:3]  # 不管影像采用的是RAS坐标系还是别的坐标系，只要有仿射矩阵，即可转换到世界坐标系（定义是唯一的,x轴正方向向右,y轴正方向向前,z正方向向上），因此世界坐标系可以作为其他坐标系间转换的中介
                        image_points=np.round(image_points).astype(int)  # 相当于最近邻插值了，因为局部影像要和局部分割的插值方式保持一致
                        in_range_index=np.where((image_points[:,0]>0)&(image_points[:,0]<y_size)&(image_points[:,1]>0)&(image_points[:,1]<z_size)&(image_points[:,2]>0)&(image_points[:,2]<x_size))  # 其余在真实影像外的元素默认为零
                        local_image=np.zeros(x_internal.ravel().shape[0],dtype='i2')  # T1影像的数据类型是int16
                        local_image[in_range_index]=img_buffer[image_points[in_range_index][:,0],image_points[in_range_index][:,1],image_points[in_range_index][:,2]]
                        local_image_reshape=local_image.reshape(x_internal.shape)  # 与ravel拆分的顺序是对应的,位置对应，只不过原点由中心平移到了左下角,图像中空气部分值可能也不是零
                        origin_img=np.array([-int(local_block_x/2), -int(local_block_y/2), -int(local_block_z), 1])
                        origin_img = origin_img[:, None]
                        real_origin_img=np.dot(coil_matrix,origin_img)
                        local_image_matrix=coil_matrix.copy()
                        local_image_matrix[:3,-1]=np.array([real_origin_img[0][0],real_origin_img[1][0],real_origin_img[2][0]])  # 为正确对应采样块图像的体素空间和世界坐标系，将原点从采样中心即采样块的上表面中心平移到采样块下表面的左下角（体素空间下的原点）
                        local_image_reshape=local_image_reshape[:,:,::-1]
                        np.save(os.path.join(local_img_save_path, subject_id + '_' + str(i) + '_' + str(theta_d) + '.npy'),local_image_reshape)
                        affine_matrices[subject_id + '_' + str(i) + '_' + str(theta_d)]=local_image_matrix.tolist()  # json不支持直接存储numpy array
                json.dump(affine_matrices,f)


if __name__=='__main__':
    main()
