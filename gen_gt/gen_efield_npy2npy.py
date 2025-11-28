import os
import argparse
import json
import numpy as np
import nibabel as nib
from multiposition_efield_cal import *
from simnibs.mesh_tools.mesh_io import read_msh, ElementData
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'


distance = 4.  # 线圈到头皮的距离，单位是mm
target_size =5  # 当custom_region为None时，该变量起作用，表示靶区半径
search_radius = 5  # 网格法搜索线圈最优放置位置时的搜索半径，单位是mm
spatial_resolution = 2.5  # 网格法搜索线圈最优放置位置时的网格边长，单位是mm
search_angle = 360  # 网格法搜索线圈最优放置朝向时的搜索角度范围，单位是°
angle_resolution = 45  # 网格法搜索线圈最优放置朝向时的搜索角度间隔，单位是°
didt = 1e6  # 线圈的dI/dt，单位是A/s
solver_options = 'pardiso'  # 选用的迭代求解器，有'pardiso'和'petsc'两种
topk = 10  # 取前topk个靶点处电场强度对应的线圈位置
num_workers = 1  # 进程数，目前多进程仅支持在Linux系统下
# opt_method = 'adm'


def main(input_folder_path,output_folder_path,target_centre,custom_region,custom_weights,custom_weights_path,coil_path,mesh2nii_path,coil_matrices):
    mesh_path=os.path.join(input_folder_path,input_folder_path.split('m2m_')[-1]+'.msh')
    reference_path=os.path.join(input_folder_path,'T1_origin_resample_noneck.nii.gz')
    # reference_path = os.path.join(input_folder_path, 'T1.nii.gz')
    interpolate_to_volume(mesh_path,reference_path,mesh2nii_path)
    tms_opt = TMSoptimize()
    tms_opt.subpath = input_folder_path
    tms_opt.pathfem =  output_folder_path
    tms_opt.fnamecoil = coil_path
    tms_opt.distance = distance
    tms_opt.search_radius = search_radius
    tms_opt.spatial_resolution = spatial_resolution
    tms_opt.search_angle = search_angle
    tms_opt.angle_resolution = angle_resolution
    tms_opt.didt = didt
    tms_opt.solver_options = solver_options
    tms_opt.target=target_centre
    tms_opt.pos_matrices=coil_matrices
    # tms_opt.method=opt_method
    opt_pos=tms_opt.run(cpus=num_workers, save_mat=False, return_n_max=topk,custom_region=custom_region,
                        custom_weights=custom_weights,custom_weights_path=custom_weights_path,mesh2nii_path=mesh2nii_path)
    opt_pos_path_matrix = os.path.join(output_folder_path, 'coil_position.npy')
    np.save(opt_pos_path_matrix,opt_pos)
    del tms_opt


def interpolate_to_volume(mesh_path, reference_path, mesh2nii_path):
    '''
    Interpolates the fields in a mesh and writem them to nifti files
    '''
    # import pdb
    # pdb.set_trace()
    mesh = read_msh(mesh_path)
    image = nib.load(reference_path)
    affine = image.affine
    n_voxels = image.header['dim'][1:4]
    # mesh = mesh.crop_mesh(elm_type=4)  # 三角形面片在前，只取四面体索引会出错
    field = np.zeros(mesh.elm.nr, dtype=np.int32)
    # field = mesh.nodes.node_number  # mesh.elm.elm_number与mesh.nodes.node_number结果不一样，一个是四面体数，一个是结点数
    field = mesh.elm.elm_number
    ed = ElementData(field)
    ed.mesh = mesh
    ed.to_nifti(n_voxels, affine, fn=mesh2nii_path, qform=image.header.get_qform(),
                method='assign')  # 采用assign方法大概
    # ed.to_nifti(n_voxels, affine, fn=mesh2nii_path, qform=image.header.get_qform(),
    #             method='linear')  # 采用assign方法大概


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='tms optimization')
    parser.add_argument('--input_folder_path', help="the path of input folder")
    parser.add_argument('--target', nargs='+', type=float, default=[], help='the pre-target coordinate in subject space')
    parser.add_argument('--target_files_path', default='', help="the path of pre-target folder")
    parser.add_argument('--custom_region', default=None,help="the style of custom region")
    parser.add_argument('--custom_weights', default=False, help="whether to load custom weights")
    parser.add_argument('--custom_weights_path', default='', help='the path of custom weights folder')
    parser.add_argument('--coil_path', default=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/simnibs_installer/simnibs/simnibs_env/lib/python3.9/site-packages/simnibs/resources/coil_models/legacy_and_other/Magstim_70mm_Fig8.ccd', help='the path of coil model')
    parser.add_argument('--coil_matrices_path', help='the path of all coil matrices')
    parser.add_argument('--coil_center_path', help='the path of coil center')
    args = parser.parse_args()
    input_folder_path=args.input_folder_path
    target=args.target
    target_files_path = args.target_files_path
    custom_region = args.custom_region
    custom_weights=args.custom_weights
    custom_weights_path=args.custom_weights_path
    coil_path=args.coil_path
    coil_matrices_path=args.coil_matrices_path
    coil_center_path=args.coil_center_path
    with open(coil_matrices_path,'r') as f1:
        coil_matrices=json.load(f1)  # 实际上这个coil matrices还不是真正的线圈参数矩阵，因为平移参数不对
    coil_center=np.load(coil_center_path)
    i=0
    num_angle=12 # 每个刺激位点的角度数
    output_dir=os.path.join(input_folder_path, 'mesh2nii_index_new_correct_interpolate_1')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    true_coil_matrices_path=os.path.join(output_dir,'coil_matrices.json')
    mesh2nii_index_path=os.path.join(output_dir,'mesh2nii_index.nii.gz')
    with open(true_coil_matrices_path, 'w') as f2:
        for k,v in coil_matrices.items():
            coil_matrices[k][0][3] = coil_center[i // num_angle][0]
            coil_matrices[k][1][3] = coil_center[i // num_angle][1]
            coil_matrices[k][2][3] = coil_center[i // num_angle][2]
            i=i+1
        json.dump(coil_matrices, f2)  # 这里的线圈矩阵实际上还是不对的,z轴正方向是垂直于头皮向外的,理论上应该是垂直于头皮向内的,在simulation/fem_me.py中有改
    output_folder_path=os.path.join(output_dir, 'efield_calculation')
    if not os.path.join(output_folder_path):
        os.makedirs(output_folder_path)
    main(input_folder_path, output_folder_path, target, custom_region=None,
         custom_weights=False, custom_weights_path='',coil_path=coil_path,mesh2nii_path=mesh2nii_index_path,coil_matrices=coil_matrices)
