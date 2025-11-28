# coding=utf-8
import numpy as np
import os
import simnibs
import csv
import scipy.spatial
from simnibs.mesh_tools import mesh_io
from tqdm import tqdm


delete_electrode=['Nz','F9','FT9','T9','TP9','F10','FT10','T10','TP10','LPA','RPA']
edge_electrode=['Fpz','Fp1','AF7','F7','FT7','T7','TP7','P9','PO9','I1','Iz','Fp2','AF8','F8','FT8','T8','TP8','P10','PO10','I2']
sub_edge_electrode=['F5','FC5','C5','CP5','PO7','O1','Oz','F6','FC6','C6','CP6','PO8','O2']
front_electrode=['AF3','AF4','AFz']  # 针对'm2m_A8','m2m_A16','m2m_A18','m2m_A24','m2m_A25','m2m_A26','m2m_B4','m2m_TMS-015','m2m_TMS-040'


def _create_grid(mesh, pos, distance, radius, resolution_pos, scalp_normals_smoothing_steps=20):
    ''' Creates a position grid '''
    # extract ROI
    msh_surf = mesh.crop_mesh(elm_type=2)
    msh_skin = msh_surf.crop_mesh([5, 1005])
    target_skin=pos
    elm_center = msh_skin.elements_baricenters()[:]
    elm_mask_roi = np.linalg.norm(elm_center - target_skin, axis=1) < 1.2 * radius  # 找到皮肤靶点方圆1.2*r范围内的element
    elm_center_zeromean = (
        elm_center[elm_mask_roi] -
        np.mean(elm_center[elm_mask_roi], axis=0)
    )
    msh_roi = msh_skin.crop_mesh(elements=msh_skin.elm.elm_number[elm_mask_roi])

    # tangential plane of target_skin point
    u, s, vh = np.linalg.svd(elm_center_zeromean)  # vh是3*3矩阵，s是3个奇异值
    vh = vh.transpose()

    # define regular grid and rotate it to head space
    coords_plane = np.array(
        np.meshgrid(
            np.linspace(-radius, radius, int(2 * radius / resolution_pos + 1)),
            np.linspace(-radius, radius, int(2 * radius / resolution_pos + 1)),
        )
    ).T.reshape(-1, 2)
    coords_plane = coords_plane[np.linalg.norm(coords_plane, axis=1) <= radius] # 由于正方形的方向不确定且会根据线段与三角形是否相交筛掉外围的一些点，因此为圆形，半径大一点即可
    coords_plane = np.dot(coords_plane, vh[:, :2].transpose()) + target_skin  # vh负责调整矩阵的方向

    # project grid-points to skin surface
    coords_mapped = []
    coords_normals = []
    normals_roi = msh_roi.triangle_normals(smooth=scalp_normals_smoothing_steps)
    q1 = coords_plane + 1e2 * resolution_pos * vh[:, 2]
    q2 = coords_plane - 1e2 * resolution_pos * vh[:, 2]
    if not q1.size and not q2.size:
        raise ValueError(f"Couldn't determine valid coil positions within search radius. Search radius too small?")
    idx, pos = msh_roi.intersect_segment(q1, q2)
    for i, c in enumerate(coords_plane):
        intersections = idx[:, 0] == i
        if np.any(intersections):
            intersect_pos = pos[intersections]
            intersect_triangles = idx[intersections, 1]  # idx第二个维度的第二个值表示三角形的索引
            dist = np.linalg.norm(c[None, :] - intersect_pos, axis=1)
            closest = np.argmin(dist)
            coords_normals.append(normals_roi[intersect_triangles[closest]])
            coords_mapped.append(intersect_pos[closest])
    coords_mapped = np.array(coords_mapped)  # 映射到头皮上的点
    # coords_normals = np.array(coords_normals)
    # coords_mapped_temp = coords_mapped + distance * coords_normals
    # coords_mapped_refine=coords_mapped_temp[(np.linalg.norm(coords_mapped,axis=1)-np.linalg.norm(coords_mapped_temp,axis=1))<-0.5*distance,:]  # 三角形两边之差小于第三边，按理说阈值是0，但考虑到有些coords_normals求解得有问题，尽可能得筛掉异常点,因此阈值设为-0.5*distance
    # return coords_mapped_refine
    return coords_mapped


def main():
    # root_path=r'G:\zhoujunfeng_g\data\cohort_lab_cognitive_impairment_5'
    # distance=4
    root_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    distance = 0  # 若resolution_pos固定,distance越小，采样点越少
    radius=28
    radius_sub_edge=32
    radius_front=22  # 针对'm2m_A8','m2m_A16','m2m_A18','m2m_A24','m2m_A25','m2m_A26', 'm2m_B4', 'm2m_TMS-015','m2m_TMS-040'
    resolution_pos=4
    min_distance=resolution_pos/2
    # foldfiles=os.listdir(root_path)
    # foldfiles=['m2m_A1','m2m_A2','m2m_A9','m2m_A11','m2m_A22']
    # foldfiles=['m2m_A1','m2m_A2']
    # foldfiles = ['m2m_A3', 'm2m_A4', 'm2m_A5', 'm2m_A6', 'm2m_A7',  'm2m_A10', 'm2m_A12', 'm2m_A13', 'm2m_A14',
    #              'm2m_A15',  'm2m_A17',  'm2m_A19', 'm2m_A20', 'm2m_A21', 'm2m_A23',
    #               'm2m_A27', 'm2m_A28', 'm2m_A29']
    # foldfiles=['m2m_A8','m2m_A16','m2m_A18','m2m_A24','m2m_A25','m2m_A26']
    # foldfiles=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
    # foldfiles = ['m2m_B4']
    foldfiles = ['m2m_TMS-015','m2m_TMS-040']
    # foldfiles = ['m2m_TMS-076','m2m_TMS-103']
    for foldfile in tqdm(foldfiles):
        subject_path=os.path.join(root_path,foldfile)
        if os.path.isdir(subject_path):
            subject_name=foldfile.split('_')[-1]
            mesh_path = os.path.join(subject_path,subject_name+'.msh')
            mesh = mesh_io.read(mesh_path)
            csv_path = os.path.join(subject_path,'eeg_positions/EEG10-10_UI_Jurak_2007.csv')
            # geo_dir = os.path.join(subject_path,'local_sampling')
            geo_dir = os.path.join(subject_path, 'local_sampling_scalp_labels_new')
            if not os.path.exists(geo_dir):
                os.makedirs(geo_dir)
            geo_path = os.path.join(geo_dir,'coil_positions_scalp.geo')
            sampling_coords_path = os.path.join(geo_dir,'coil_positions.npy')
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader=csv.reader(f)
                index=0
                for row in reader:
                    if row[4] not in delete_electrode and row[4] not in edge_electrode:
                        index=index+1
                        pos=[float(row[1]),float(row[2]),float(row[3])]
                        if row[4] in sub_edge_electrode:
                            coords_mapped = _create_grid(mesh, pos, distance, radius_sub_edge, resolution_pos,scalp_normals_smoothing_steps=20)
                        elif row[4] in front_electrode:  # 针对'm2m_A8','m2m_A16','m2m_A18','m2m_A24','m2m_A25','m2m_A26','m2m_B4', 'm2m_TMS-015','m2m_TMS-040'
                            coords_mapped = _create_grid(mesh, pos, distance, radius_front, resolution_pos, scalp_normals_smoothing_steps=20)  # 针对'm2m_A8','m2m_A16','m2m_A18','m2m_A24','m2m_A25','m2m_A26','m2m_B4','m2m_TMS-015'
                        else:
                            coords_mapped=_create_grid(mesh, pos, distance, radius, resolution_pos, scalp_normals_smoothing_steps=20)
                        if index==1:
                            coords_set=coords_mapped
                        else:
                            coords_set=np.concatenate((coords_set,coords_mapped),axis=0)
            diff = coords_set[:, np.newaxis, :] - coords_set[np.newaxis, :, :]
            dist_squared = np.sum(diff ** 2, axis=2)  # n*n的矩阵
            np.fill_diagonal(dist_squared, np.inf)
            mask = dist_squared >= min_distance ** 2
            upper_tri_indices = np.triu_indices_from(mask, k=1)  # upper_tri_indices返回两个array，第一个array存储上三角元素的行，第二个array上三角元素的列
            to_remove_indices = upper_tri_indices[0][np.where(~mask[upper_tri_indices])[0]]  # 找到mask中上三角部分False元素对应的行，这样能保证到时删掉序号靠前的元素
            to_remove_indices = np.unique(to_remove_indices)  # 去重并排序，因为同一个点可能与多个点都过近
            filtered_coords_set = np.delete(coords_set, to_remove_indices, axis=0)  # 删掉对应的元素
            with open(geo_path, 'w') as f:
                f.write("View 'grids' {\n")
                for coords in filtered_coords_set:
                    f.write(
                        "SP(" + ", ".join([str(i) for i in coords]) + ")"
                        "{1};\n")
                f.write("};\n")
            np.save(sampling_coords_path, filtered_coords_set)

if __name__=='__main__':
    main()
