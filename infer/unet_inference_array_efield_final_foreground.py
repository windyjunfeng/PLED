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
from networks.unet_true_1_fusion_1 import UNet_true_1_fusion_1


def inference_efield(temp_img_path,weight_path,dAdt_file,output_path):
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
    # device = torch.device("cpu")
    model = UNet_true_1_fusion_1(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(8, 16, 32, 64, 128),
        strides=(1, 2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(weight_path,map_location=device))
    model.eval()
    dAdt = np.load(dAdt_file)
    dAdt = torch.tensor(dAdt).to(device)
    # conductivity_map = {'1': 0.126, '2': 0.275, '3': 1.654, '4': 0.01, '5': 0.465, '9': 0.6}
    conductivity_map={'1': 0.126, '2': 0.275, '3': 1.654, '4': 0.01, '5': 0.465, '9': 0.6, '11': 0.108, '12': 0.237, '13': 1.422, '14': 0.009, '15': 0.400, '19': 0.516}
    # import pdb
    # pdb.set_trace()
    with tqdm(total=len(test_imgs)) as pbar:
        with torch.no_grad():
            for i in range(len(test_imgs)):
                test_img = test_imgs[i]
                test_img_array = np.load(test_img)
                test_img_tensor = torch.tensor(test_img_array).float().to(device)
                test_img_tensor = (test_img_tensor.unsqueeze(0)).unsqueeze(0)
                for k in conductivity_map:
                    test_img_tensor=torch.where(test_img_tensor==int(k),torch.tensor(conductivity_map[k]).to(torch.float32).to(device),test_img_tensor)
                dAdt_val = dAdt.unsqueeze(0).repeat(test_img_tensor.shape[0], 1, 1, 1, 1)
                dAdt_mask = torch.where(test_img_tensor == 0, torch.tensor(0).to(torch.float32).to(device), dAdt_val)
                test_output = model(test_img_tensor, dAdt_mask)
                test_output_foreground = test_output.masked_fill(~(test_img_tensor != 0), 0)  # 背景部分直接置为零
                test_output_foreground = torch.squeeze(test_output_foreground)
                test_output_foreground_array=test_output_foreground.to('cpu').numpy().astype('float16')
                np.save(os.path.join(output_path,files[i]),test_output_foreground_array)
                pbar.update(1)


if __name__ == "__main__":
    # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label/m2m_A9'
    # temp_img_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
    temp_img_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    weight_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/code/deep_learning_e-field/segmentation/unet/39_best_metric_model_efieldcal_array_fusion_1_new_interpolate_all.pth'
    dAdt_file = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/dadt_norm_808032_0mm_fmm3d.npy'
    # output_path=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield_inference_new_correct_conductivity/m2m_A9'
    # inference_efield(temp_img_path,weight_path,dAdt_file,output_path)
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/inference/local_efield_final_based_gtlabel_best'
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/inference/local_efield_final_based_step1_best_temp'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_temp.txt'
    output_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    split_test_txt = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/test.txt'
    f1 = open(split_test_txt, 'r')
    lines = f1.readlines()
    for line in lines:
        # temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'local_sampling_scalp_labels_new/local_label')
        temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'inference','local_seg_final')
        output_path = os.path.join(output_rootpath,line.rstrip('\n'),'inference','local_efield_final_based_step1')
        inference_efield(temp_img_path, weight_path, dAdt_file, output_path)
