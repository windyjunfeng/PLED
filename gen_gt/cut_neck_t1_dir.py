import os
import nibabel as nib
import numpy as np
import scipy.ndimage.morphology as mrph
from tqdm import tqdm
from scipy.io import loadmat
from simnibs.utils.transformations import volumetric_affine


def cut_neck(fn_mni_template,fn_affine,image_path,image_noneck_path,n_dil):
    mni_image = nib.load(fn_mni_template)
    mni_buffer = np.ones(mni_image.shape, dtype=bool)
    mni_affine = mni_image.affine  # MNI模板影像认为是源，original space下从体素空间到世界坐标系的转换矩
    image=nib.load(image_path)
    image_buffer = np.round(image.get_fdata()).astype(np.int16)
    image_affine = image.affine
    trafo = loadmat(fn_affine)['worldToWorldTransformMatrix']
    upperhead = volumetric_affine((mni_buffer, mni_affine),
                                      np.linalg.inv(trafo),  # 个体影像的世界坐标系到MNI模板的世界坐标系转换矩阵
                                      target_space_affine=image_affine,
                                      target_dimensions=image.shape,
                                      intorder=0)
    upperhead = mrph.binary_dilation(upperhead, iterations=n_dil)
    image_buffer[~upperhead] = 0
    image_noneck = nib.Nifti1Image(image_buffer, image_affine)
    nib.save(image_noneck, image_noneck_path)


if __name__ == '__main__':
    root_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    fn_mni_template = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/simnibs_installer/simnibs/simnibs_env/lib/python3.9/site-packages/simnibs/resources/templates/MNI152_T1_1mm.nii.gz'
    n_dil = 30  # 体素间距为1mm时
    subjects=os.listdir(root_path)
    for subject in tqdm(subjects):
        if 'm2m_' not in subject:
            continue
        fn_affine = os.path.join(root_path,subject,r'segmentation/coregistrationMatrices.mat')  # MNI模板影像与原T1影像间的坐标转换矩阵
        image_path = os.path.join(root_path,subject,'T1_origin_resample.nii.gz')
        image_noneck_path = os.path.join(root_path,subject,'T1_origin_resample_noneck.nii.gz')
        cut_neck(fn_mni_template,fn_affine,image_path,image_noneck_path,n_dil)
