# coding=utf-8
import os
import numpy as np


temp_img_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
split_test_txt=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/test.txt'
output_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/evaluation/val_filename_order.npy'
f1 = open(split_test_txt, 'r')
lines_1 = f1.readlines()
test_imgs=[]
for line in lines_1:
    temp_img_path_whole = os.path.join(temp_img_path, line.rstrip('\n'),'local_sampling_scalp_imgs_new/local_image')
    filenames = os.listdir(temp_img_path_whole)
    for filename in filenames:
        test_imgs.append(filename)
test_imgs=np.array(test_imgs)
np.save(output_path,test_imgs)