# coding=utf-8
import os
import numpy as np
import nibabel.gifti as gif
import nibabel as nib


if __name__=="__main__":
    root_path = r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\visualization'
    test_subject=['m2m_A2']
    subject_index=0
    # sampling_list=['A2_311_45','A2_311_-45']
    # sampling_list = ['A2_1781_45', 'A2_1781_-45']
    # sampling_list = ['A2_1488_45', 'A2_1488_-45']
    sampling_list = ['A2_1488_-45']
    sampling_index=0
    for sampling_index in range(len(sampling_list)):
        sub_path=os.path.join(root_path,test_subject[subject_index],sampling_list[sampling_index])
        gm_data=nib.load(os.path.join(sub_path,sampling_list[sampling_index]+'_gm.surf.gii'))
        vertices=gm_data.darrays[0].data
        zero_color=np.zeros(vertices.shape[0], dtype=np.float32)
        gii_file = gif.GiftiImage()
        gii_file.add_gifti_data_array(gif.GiftiDataArray(zero_color))
        output_gii_path=os.path.join(sub_path,sampling_list[sampling_index]+'_zero_color.shape.gii')
        nib.save(gii_file, output_gii_path)