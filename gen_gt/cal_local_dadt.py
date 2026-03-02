import os
import time
import numpy as np
import nibabel as nib


try:
    import fmm3dpy
except ImportError:
    FMM3D = False
else:
    FMM3D = True


def set_up_tms_dAdt(local_block, coil_file, coil_matrix, didt=1e6):
    coil_matrix = np.array(coil_matrix)
    if isinstance(coil_file, nib.nifti1.Nifti1Image) or\
        coil_file.endswith('.nii.gz') or\
        coil_file.endswith('.nii'):
        dadt = _calculate_dadt_nifti(local_block, coil_file,
                                     coil_matrix, didt)
    elif coil_file.endswith('.ccd'):
        if FMM3D:
            dadt = _calculate_dadt_ccd_FMM(
                local_block, coil_file,
                coil_matrix, didt
            )
        else:
            dadt = _calculate_dadt_ccd(
                local_block, coil_file,
                coil_matrix, didt
            )
    else:
        raise ValueError('coil file must be either a .ccd file or a nifti file')
    return dadt


def _rotate_coil(ccd_file, coil_matrix):
    # read ccd file
    d_position, d_moment = read_ccd(ccd_file)
    # transfrom positions to mm
    d_position *= 1e3
    # add a column to the position in order to apply the transformation matrix
    d_position = np.hstack([d_position, np.ones((d_position.shape[0], 1))])
    d_position = coil_matrix.dot(d_position.T).T[:, :3]
    # rotate the moment
    d_moment = coil_matrix[:3, :3].dot(d_moment.T).T
    return d_position, d_moment

def _calculate_dadt_ccd(local_block, ccd_file, coil_matrix, didt):
    """ auxiliary function to calculate the dA/dt field from a ccd file """
    # read ccd file
    # import pdb
    # pdb.set_trace()
    d_position, d_moment = _rotate_coil(ccd_file, coil_matrix)
    A = np.zeros((len(local_block), 3), dtype=float)
    for p, m in zip(d_position, d_moment):
        # get distance of point to dipole, transform back to meters
        r = (local_block - p) * 1e-3
        A += 1e-7 * didt * np.cross(m, r) / (np.linalg.norm(r, axis=1)[:, None] ** 3)
    return A

def _calculate_dadt_ccd_FMM(local_block, ccd_file, coil_matrix, didt, eps=1e-3):
    """ auxiliary function to calculate the dA/dt field from a ccd file using FMM """
    d_position, d_moment = _rotate_coil(ccd_file, coil_matrix)
    # bring everything to SI
    d_position *= 1e-3
    pos = local_block * 1e-3
    A = np.zeros((len(pos), 3), dtype=float)
    out = fmm3dpy.lfmm3d(
            eps=eps,
            sources=d_position.T,
            charges=d_moment.T,
            targets=pos.T,
            pgt=2,
            nd=3
        )
    A[:, 0] = (out.gradtarg[1][2] - out.gradtarg[2][1])
    A[:, 1] = (out.gradtarg[2][0] - out.gradtarg[0][2])
    A[:, 2] = (out.gradtarg[0][1] - out.gradtarg[1][0])
    A *= -1e-7 * didt
    return A


def _calculate_dadt_nifti(local_block, nifti_image, coil_matrix, didt):
    """ auxiliary function that interpolates the dA/dt field from a nifti file """
    if isinstance(nifti_image, str):
        nifti_image = nib.load(nifti_image)
    elif isinstance(nifti_image, nib.nifti1.Nifti1Image):
        pass
    else:
        raise NameError('Failed to parse input volume (not string or nibabel nifti1 volume)')
    coords = local_block
    out = _get_field(nifti_image, coords, coil_matrix)
    out = out * didt
    return out.T

def read_ccd(fn):
    """ reads a ccd file

    Parameters
    -----------
    fn: str
        name of ccd file

    Returns
    ----------
    [pos, m]: list
        position and moment of dipoles
    """
    # import pdb
    # pdb.set_trace()
    ccd_file = np.loadtxt(fn, skiprows=2)
    # if there is only 1 dipole, loadtxt return as array of the wrong shape
    if (len(np.shape(ccd_file)) == 1):
        a = np.zeros([1, 6])
        a[0, 0:3] = ccd_file[0:3]
        a[0, 3:] = ccd_file[3:]
        ccd_file = a
    return ccd_file[:, 0:3], ccd_file[:, 3:]


def _get_field(nifti_image, coords, coil_matrix, get_norm=False):
    ''' This function is also used in the GUI '''
    from scipy.ndimage import interpolation
    if isinstance(nifti_image, str):
        nifti_image = nib.load(nifti_image)
    elif isinstance(nifti_image, nib.nifti1.Nifti1Image):
        pass
    else:
        raise NameError('Failed to parse input volume (not string or nibabel nifti1 volume)')
    iM = np.dot(np.linalg.pinv(nifti_image.affine),
                np.linalg.pinv(coil_matrix))

    # gets the coordinates in voxel space
    voxcoords = np.dot(iM[:3, :3], coords.T) + iM[:3, 3][:, np.newaxis]

    # Interpolates the values of the field in the given coordinates
    out = np.zeros((3, voxcoords.shape[1]))
    for dim in range(3):
        out[dim] = interpolation.map_coordinates(
            np.asanyarray(nifti_image.dataobj)[..., dim],
            voxcoords, order=1
        )

    # Rotates the field
    out = np.dot(coil_matrix[:3, :3], out)
    if get_norm:
        out = np.linalg.norm(out, axis=0)
    return out


if __name__=="__main__":
    output_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/dadt_norm_808064_0mm_fmm3d.npy'
    # output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field\e-field_calculation\dadt_norm_808032_0mm_fmm3d_temp.npy'
    # output_path = r'G:\zhoujunfeng_g\code\deep_learning_e-field\e-field_calculation\dadt_808032_4mm.npy'
    coil_file=r'/data/disk_2/zhoujunfeng/simnibs_installer/simnibs/simnibs_env/lib/python3.9/site-packages/simnibs/resources/coil_models/legacy_and_other/Magstim_70mm_Fig8.ccd'
    # coil_file=r'D:\simnibs\simnibs_env\Lib\site-packages\simnibs\resources\coil_models\legacy_and_other\Magstim_70mm_Fig8.ccd'  # 实际上里面的偶极子的z轴坐标实际位于-4.1666和-8.8333之间，即偶极子距离头皮最近也有4mm
    # coil_file=r'D:\simnibs\simnibs_env\Lib\site-packages\simnibs\resources\coil_models\legacy_and_other\Magstim_70mm_Fig8.nii.gz'
    local_block_x = 80  # 局部采样块x轴方向上的边长
    local_block_y = 80  # 局部采样块y轴方向上的边长
    # local_block_z = 32  # 局部采样块z轴方向上的边长
    local_block_z = 64  # 局部采样块z轴方向上的边长
    distance = 0 # 线圈距头皮的距离
    # coil_matrix=[
    #     [1,0,0,0],
    #     [0,1,0,0],
    #     [0,0,1,0],
    #     [0,0,0,1]]  # 线圈紧贴头皮
    coil_matrix=[
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,-1.0*distance],
        [0,0,0,1]]
    didt=1e6

    start_time=time.time()
    x_internal, y_internal, z_internal = np.mgrid[-int(local_block_x / 2):int(local_block_x / 2),
                                             -int(local_block_y / 2):int(local_block_y / 2), 0:local_block_z:1]  # 实际上线圈的z轴正方向是垂直于头皮向内的
    internal_points = np.vstack((x_internal.ravel(),y_internal.ravel(),z_internal.ravel())).T
    dadt=set_up_tms_dAdt(internal_points, coil_file, coil_matrix, didt)
    dadt=dadt.reshape(x_internal.shape[0],x_internal.shape[1],x_internal.shape[2],3).astype(np.float32)
    dadt = dadt[:, :, ::-1, :]  # 线圈的z轴正方向和img的z轴正方向正好相反，这里的正方向为了和img匹配，经过此步后z轴正方向垂直于头皮向外
    # np.save(output_path, dadt)

    dadt_norm=np.linalg.norm(dadt,axis=3)
    np.save(output_path, dadt_norm)

    end_time=time.time()
    runtime=end_time-start_time
    print(f'程序运行时间： {runtime:.4f} 秒')