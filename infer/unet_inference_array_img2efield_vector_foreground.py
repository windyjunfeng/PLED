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


def inference_efield_vector(temp_img_path,temp_seg_path, weight_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num_classes = 3
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    test_imgs=[]
    test_segs=[]
    files = os.listdir(temp_img_path)
    for file in files:
        test_imgs.append(os.path.join(temp_img_path, file))
        test_segs.append(os.path.join(temp_seg_path, file))
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
                test_seg = torch.tensor(np.load(test_segs[i]))[None, None, :].to(device)
                test_output_foreground = test_output.masked_fill(~(test_seg != 0), 0)  # 背景部分直接置为零
                test_output_foreground = torch.squeeze(test_output_foreground)
                test_output_foreground_array=test_output_foreground.permute(1, 2, 3, 0).to('cpu').numpy().astype('float16')
                np.save(os.path.join(output_path,files[i]),test_output_foreground_array)
                pbar.update(1)


if __name__ == "__main__":
    temp_img_rootpath = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    temp_seg_rootpath = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    weight_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/weights/45_best_metric_model_img2efieldcal_vector_array_true_1_interpolate.pth'
    output_rootpath = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    split_test_txt = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/val_l.txt'
    f1 = open(split_test_txt, 'r')
    lines = f1.readlines()
    for line in lines:
        temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'local_sampling_scalp_imgs_new_1010_center/local_image')
        temp_seg_path = os.path.join(temp_seg_rootpath, line.rstrip('\n'),'local_sampling_scalp_labels_new_1010_center/local_label')
        # output_path = os.path.join(output_rootpath,line.rstrip('\n'))
        output_path = os.path.join(output_rootpath, line.rstrip('\n'),'inference/local_img2efield_vector_new_1010_center')
        inference_efield_vector(temp_img_path, temp_seg_path, weight_path, output_path)
    
    # temp_img_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/m2m_A9/local_sampling_scalp_imgs_new_1010_center/local_image'
    # temp_seg_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/m2m_A9/local_sampling_scalp_labels_new_1010_center/local_label'
    # weight_path = r'/data/disk_2/zhoujunfeng/code/efield_calculation/weights/45_best_metric_model_img2efieldcal_vector_array_true_1_interpolate.pth'
    # output_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/m2m_A9/inference/local_img2efield_vector_new_1010_center'
    # inference_efield_vector(temp_img_path, temp_seg_path, weight_path, output_path)