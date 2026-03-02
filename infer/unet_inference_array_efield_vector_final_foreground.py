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
from networks.unet_true_1_fusion_1_dual_channels import UNet_true_1_fusion_1_dual_channels


def inference_efield_vector(temp_img_path, weight_path, dAdt_file, output_path):
    """
    功能：使用训练好的模型预测矢量电场（3个分量：x, y, z）
    :param temp_img_path: 输入图像路径（包含.npy文件的目录）
    :param weight_path: 模型权重文件路径
    :param dAdt_file: dAdt文件路径（numpy数组，shape: (H, W, D, 3)）
    :param output_path: 输出路径
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num_classes = 3  # 预测电场矢量分量（x, y, z）
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    test_imgs = []
    files = os.listdir(temp_img_path)
    for file in files:
        test_imgs.append(os.path.join(temp_img_path, file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    # 使用dual_channels模型，支持双输入（图像和dAdt）
    model = UNet_true_1_fusion_1_dual_channels(
        spatial_dims=3,
        in_channels_1=1,  # 图像输入通道数
        in_channels_2=3,  # dAdt输入通道数
        out_channels=num_classes,  # 输出3个通道（x, y, z分量）
        channels=(8, 16, 32, 64, 128),
        strides=(1, 2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    # 加载dAdt文件，shape: (H, W, D, 3)
    dAdt = np.load(dAdt_file)
    dAdt = torch.tensor(dAdt).to(device)
    
    conductivity_map = {'1': 0.126, '2': 0.275, '3': 1.654, '4': 0.01, '5': 0.465, '9': 0.6}
    # conductivity_map={'1': 0.126, '2': 0.275, '3': 1.654, '4': 0.01, '5': 0.465, '9': 0.6, '11': 0.108, '12': 0.237, '13': 1.422, '14': 0.009, '15': 0.400, '19': 0.516}
    
    with tqdm(total=len(test_imgs)) as pbar:
        with torch.no_grad():
            for i in range(len(test_imgs)):
                test_img = test_imgs[i]
                test_img_array = np.load(test_img)
                test_img_tensor = torch.tensor(test_img_array).float().to(device)
                test_img_tensor = (test_img_tensor.unsqueeze(0)).unsqueeze(0)  # (1, 1, H, W, D)
                
                # 应用conductivity_map转换
                for k in conductivity_map:
                    test_img_tensor = torch.where(
                        test_img_tensor == int(k),
                        torch.tensor(conductivity_map[k]).to(torch.float32).to(device),
                        test_img_tensor
                    )
                
                # dAdt shape: (H, W, D, 3) -> (3, H, W, D) -> (batch, 3, H, W, D)
                dAdt_val = dAdt.permute(3, 0, 1, 2).unsqueeze(0).repeat(test_img_tensor.shape[0], 1, 1, 1, 1)
                # 利用分割结果作掩码，背景部分dAdt置为0
                dAdt_mask = torch.where(
                    test_img_tensor == 0,
                    torch.tensor(0).to(torch.float32).to(device),
                    dAdt_val
                )
                
                # 模型推理：输入图像和dAdt，输出矢量电场（3个通道）
                test_output = model(test_img_tensor, dAdt_mask)  # (batch, 3, H, W, D)
                
                # 背景部分直接置为零（只保留前景区域的预测结果）
                test_output_foreground = test_output.masked_fill(~(test_img_tensor != 0), 0)
                
                # 移除batch维度
                test_output_foreground = torch.squeeze(test_output_foreground)  # (3, H, W, D)
                
                # 转换为numpy数组，并转置为(H, W, D, 3)格式以便保存
                test_output_foreground_array = test_output_foreground.permute(1, 2, 3, 0).to('cpu').numpy().astype('float16')  # (H, W, D, 3)
                
                # 保存为npy文件
                np.save(os.path.join(output_path, files[i]), test_output_foreground_array)
                pbar.update(1)


if __name__ == "__main__":
    # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label/m2m_A9'
    # temp_img_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
    temp_img_rootpath = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    weight_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/weights/42_4_best_metric_model_efieldcal_vector_array_fusion_1_dual_channels.pth'
    dAdt_file = r'/data/disk_2/zhoujunfeng/code/efield_calculation/dadt_808032_0mm_fmm3d.npy'
    # output_path=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield_inference_new_correct_conductivity/m2m_A9'
    # inference_efield(temp_img_path,weight_path,dAdt_file,output_path)
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/inference/local_efield_final_based_gtlabel_best'
    # output_rootpath = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/inference/local_efield_final_based_step1_best_temp'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_temp.txt'
    output_rootpath = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    split_test_txt = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/val_l.txt'
    f1 = open(split_test_txt, 'r')
    lines = f1.readlines()
    for line in lines:
        # temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'local_sampling_scalp_labels_new_1010_center/local_label')
        temp_img_path = os.path.join(temp_img_rootpath, line.rstrip('\n'),'inference','local_seg_final_new_1010_center')
        output_path = os.path.join(output_rootpath,line.rstrip('\n'),'inference','local_efield_vector_final_new_1010_center_based_step1')
        inference_efield_vector(temp_img_path, weight_path, dAdt_file, output_path)
    
    # temp_img_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/m2m_A9/local_sampling_scalp_labels_new_1010_center/local_label'
    # weight_path = r'/data/disk_2/zhoujunfeng/code/efield_calculation/weights/42_4_best_metric_model_efieldcal_vector_array_fusion_1_dual_channels.pth'
    # dAdt_file = r'/data/disk_2/zhoujunfeng/code/efield_calculation/dadt_808032_0mm_fmm3d.npy'
    # output_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/m2m_A9/inference/local_efield_vector_final_new_1010_center'
    # inference_efield_vector(temp_img_path, weight_path, dAdt_file, output_path)
