# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from monai import config
from networks.unet_true_1 import UNet_true_1


def inference_efield(temp_img_path, weight_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num_classes = 1
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    test_imgs=[]
    files = os.listdir(temp_img_path)
    for file in files:
        test_imgs.append(os.path.join(temp_img_path, file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_true_1(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(8, 16, 32, 64, 128),
        strides=(1, 2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(weight_path,map_location=device))
    model.eval()
    # import pdb
    # pdb.set_trace()
    with tqdm(total=len(test_imgs)) as pbar:
        with torch.no_grad():
            for i in range(len(test_imgs)):
                test_img = test_imgs[i]
                test_img_array = np.load(test_img)
                test_img_tensor = torch.tensor(test_img_array).float().to(device)
                test_img_tensor = (test_img_tensor.unsqueeze(0)).unsqueeze(0)
                test_output = model(test_img_tensor)
                test_output_foreground = test_output.masked_fill(~(test_img_tensor != 0), 0)
                test_output_foreground = torch.squeeze(test_output_foreground)
                test_output_foreground_array=test_output_foreground.to('cpu').numpy().astype('float16')
                np.save(os.path.join(output_path,files[i]),test_output_foreground_array)
                pbar.update(1)


if __name__ == "__main__":
    # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label/m2m_A9'
    # temp_img_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
    # temp_img_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/local_seg_final'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/31_best_metric_model_efieldcal_array_true_1_interpolate_all.pth'
    # output_path=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield_inference_new_correct_conductivity/m2m_A9'
    # inference_efield(temp_img_path,weight_path,dAdt_file,output_path)
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/inference/31_based_step1'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_l.txt'
    temp_img_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    weight_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/code/deep_learning_e-field/segmentation/unet/31_best_metric_model_efieldcal_array_true_1_interpolate_all.pth'
    output_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    split_test_txt = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/test.txt'
    f1 = open(split_test_txt, 'r')
    lines = f1.readlines()
    for line in lines:
        # temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'local_sampling_scalp_labels_new/local_label')
        # temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'))
        # output_path = os.path.join(output_rootpath,line.rstrip('\n'))
        temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'inference/local_seg_final')
        output_path = os.path.join(output_rootpath,line.rstrip('\n'),'inference/local_efield_31_based_step1')
        inference_efield(temp_img_path, weight_path, output_path)