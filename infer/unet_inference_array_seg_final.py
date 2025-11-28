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
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from monai import config
from networks.unet_true_1 import UNet_true_1
from monai.transforms.utils import rescale_array
from monai.utils.type_conversion import convert_to_dst_type


def inference_seg(temp_img_path,weight_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num_classes = 10
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    test_imgs=[]
    files = os.listdir(temp_img_path)
    for file in files:
        test_imgs.append(os.path.join(temp_img_path, file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
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
    with tqdm(total=len(test_imgs)) as pbar:
        with torch.no_grad():
            for i in range(len(test_imgs)):
                test_img=test_imgs[i]
                # import pdb
                # pdb.set_trace()
                test_img_array=np.load(test_img)
                test_img_tensor=torch.tensor(test_img_array).to(device)
                test_img_tensor=(test_img_tensor.unsqueeze(0)).unsqueeze(0)
                test_img_tensor=rescale_array(test_img_tensor, 0.0, 1.0, dtype=np.float32)
                test_output = model(test_img_tensor)
                test_output=torch.squeeze(test_output,dim=0)
                test_output=torch.argmax(test_output,dim=0)  # 在one-hot通道数中数值最大的索引
                test_output_array=test_output.to('cpu').numpy().astype('i2')
                np.save(os.path.join(output_path,files[i]),test_output_array)
                pbar.update(1)  # 更新进度条


if __name__ == "__main__":
    # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/image/m2m_A9/'
    temp_img_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/32_best_best_metric_model_segmentation3d_array_scalp_true_1_all.pth'
    # weight_path = r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/27_best_metric_model_segmentation3d_array_scalp_true_1.pth'
    weight_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/code/deep_learning_e-field/segmentation/unet/32_best_best_metric_model_segmentation3d_array_scalp_true_1_all.pth'
    # output_path=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/local_label_inference/m2m_A9'
    # inference_seg(temp_img_path,weight_path,output_path)
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/local_seg_final'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_l.txt'
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/local_seg_small'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_s.txt'
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/32_cog_cpu'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/cog_test_temp.txt'
    output_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    split_test_txt = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/test.txt'
    f1 = open(split_test_txt, 'r')
    lines = f1.readlines()
    for line in lines:
        temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'local_sampling_scalp_imgs_new/local_image')
        output_path = os.path.join(output_rootpath, line.rstrip('\n'),'inference','local_seg_final')
        inference_seg(temp_img_path, weight_path, output_path)
