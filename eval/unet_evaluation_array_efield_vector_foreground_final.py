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
import torch.multiprocessing as mp
import nibabel as nib
import numpy as np
import cucim
import torch
from tqdm import tqdm
from scipy.ndimage import binary_erosion
from monai import config
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
# from monai.networks.nets import UNet
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, SaveImage, ScaleIntensity,ResizeWithPadOrCrop, Transpose
from networks.unet_true_1_fusion_1_dual_channels import UNet_true_1_fusion_1_dual_channels
from monai.utils import convert_to_cupy,convert_to_tensor


# mp.set_sharing_strategy('file_system')


def main(temp_img_path,temp_label_path,split_test_txt,weight_path, output_dir, dAdt_file):
    num_classes = 3  # 预测电场矢量分量
    num_labels=10
    # num_labels = 20
    batch_size=32
    voxel_per_img=80*80*32
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    f1 = open(split_test_txt, 'r')
    lines_2 = f1.readlines()
    test_imgs=[]
    test_labels=[]
    # import pdb
    # pdb.set_trace()
    for line in lines_2:
        temp_img_path_whole=os.path.join(temp_img_path, line.rstrip('\n'), 'local_sampling_scalp_labels_new_1010_center/local_label')
        # temp_img_path_whole = os.path.join(temp_img_path, line.rstrip('\n'))
        # temp_label_path_whole=os.path.join(temp_label_path, line.rstrip('\n'), 'local_efield')
        temp_label_path_whole = os.path.join(temp_label_path, line.rstrip('\n'), 'mesh2nii_index_new_correct_interpolate_vector/efield_calculation', 'local_efield_voxel_coord')
        files=os.listdir(temp_img_path_whole)
        for file in files:
            test_imgs.append(os.path.join(temp_img_path_whole, file))
            test_labels.append(os.path.join(temp_label_path_whole, file))
    # define transforms for image and segmentation
    imtrans = Compose([EnsureChannelFirst()])
    segtrans = Compose([Transpose(indices=(3, 0, 1, 2))])  # 矢量标签是4D的，需要转置
    val_ds = ImageDataset(test_imgs, test_labels, transform=imtrans, seg_transform=segtrans)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())  # 将一个batch的数据拆开时这样会报错
    # val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_true_1_fusion_1_dual_channels(
        spatial_dims=3,
        in_channels_1=1,  # 图像输入通道数
        in_channels_2=3,  # dAdt输入通道数
        out_channels=num_classes,
        channels=(8, 16, 32, 64, 128),
        strides=(1, 2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(weight_path,map_location=device))
    model.eval()
    # 加载dAdt文件
    dAdt = np.load(dAdt_file)
    dAdt = torch.tensor(dAdt).to(device)
    conductivity_map = {'1':0.126,'2':0.275,'3':1.654,'4':0.01,'5':0.465,'9':0.6}
    # category_names = ['1','2','3']
    category_names = ['1', '2', '3', '4', '5','9']
    # category_names = ['9']
    index_names = ['mae','mre','mse','f_norm','pearson_corr','psnr','mae_loss_edge','mre_loss_edge','mae_loss_mean','vector_angle','vector_diff_magnitude']
    with tqdm(total=len(val_loader)) as pbar:
        with torch.no_grad():
            mae_loss = []
            mae_loss_edge = []
            mae_loss_mean = []
            mae_loss_percentile95 = []
            mae_loss_percentile99 = []
            mae_loss_100max = []
            mre_loss = []
            mre_loss_edge = []
            mse_loss = []
            f_norm = []
            pearson_corr=[]
            psnr = []
            vector_angle = []  # 矢量夹角（方向差异）  
            vector_diff_magnitude = []  # 矢量差的模长（综合评估）
            evaluation_items = {category: {index_name: [] for index_name in index_names} for category in category_names}
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                # import pdb
                # pdb.set_trace()
                val_images_origin = val_images.clone()
                # 应用conductivity_map转换
                for k in conductivity_map:
                    val_images = torch.where(val_images == int(k), torch.tensor(conductivity_map[k]).to(torch.float32).to(device), val_images)
                # dAdt shape: (H, W, D, 3) -> (3, H, W, D) -> (batch, 3, H, W, D)
                dAdt_val = dAdt.permute(3,0,1,2).unsqueeze(0).repeat(val_images.shape[0], 1, 1, 1, 1)
                dAdt_mask = torch.where(val_images == 0, torch.tensor(0).to(torch.float32).to(device), dAdt_val)  # 利用分割结果作掩码
                val_outputs = model(val_images, dAdt_mask)
                edge_mask_batch = edge_mask(val_images_origin, num_labels, device).to(device)
                for i in range(val_images.shape[0]):  # 这里不应该是batch size，因为最后一个batch不一定是满batch的
                    # 对于矢量，需要处理3个通道，val_outputs shape: (batch, 3, H, W, D)
                    # val_labels shape: (batch, 3, H, W, D)
                    # # 先计算整个空间的矢量幅值
                    # val_outputs_magnitude_full = torch.sqrt(torch.sum(val_outputs[i] ** 2, dim=0))  # (H, W, D)
                    # val_labels_magnitude_full = torch.sqrt(torch.sum(val_labels[i] ** 2, dim=0))  # (H, W, D)
                    # 提取前景区域：基于矢量幅值不为0的区域
                    foreground_mask = val_images[i] != 0
                    # 对于矢量，提取前景区域的矢量值
                    val_outputs_foreground = val_outputs[i][:, torch.squeeze(foreground_mask,dim=0)]
                    val_labels_foreground = val_labels[i][:, torch.squeeze(foreground_mask,dim=0)]
                    # 计算前景区域的矢量幅值
                    val_outputs_magnitude = torch.sqrt(torch.sum(val_outputs_foreground ** 2, dim=0)) 
                    val_labels_magnitude = torch.sqrt(torch.sum(val_labels_foreground ** 2, dim=0)) 
                    # 边缘掩码（针对前景区域）
                    edge_mask_foreground = edge_mask_batch[i][foreground_mask] 
                    # MAE: 使用矢量幅值的平均绝对误差
                    mae_loss_ = torch.mean(torch.abs(val_outputs_magnitude - val_labels_magnitude))
                    # 边缘区域的MAE（使用矢量幅值）
                    if torch.any(edge_mask_foreground):
                        mae_loss_edge_ = torch.mean(torch.abs(val_outputs_magnitude[edge_mask_foreground] - val_labels_magnitude[edge_mask_foreground]))
                    else:
                        mae_loss_edge_ = torch.tensor(float('nan')).to(device)
                    # 均值差异（使用矢量幅值）
                    mae_loss_mean_ = torch.abs(torch.mean(val_outputs_magnitude) - torch.mean(val_labels_magnitude))
                    # 百分位数（使用矢量幅值）
                    mae_loss_percentile95_ = torch.abs(torch.quantile(val_outputs_magnitude, 0.95) - torch.quantile(val_labels_magnitude, 0.95))
                    mae_loss_percentile99_ = torch.abs(torch.quantile(val_outputs_magnitude, 0.99) - torch.quantile(val_labels_magnitude, 0.99))
                    # Top 100最大值（使用矢量幅值）
                    if len(val_outputs_magnitude) >= 100:
                        mae_loss_100max_ = torch.abs(torch.topk(val_outputs_magnitude, 100)[0][-1] - torch.topk(val_labels_magnitude, 100)[0][-1])
                    else:
                        mae_loss_100max_ = torch.abs(torch.max(val_outputs_magnitude) - torch.max(val_labels_magnitude))
                    # MRE: 使用矢量幅值计算相对误差
                    mre_loss_ = torch.sum(torch.nan_to_num(torch.abs((val_outputs_magnitude - val_labels_magnitude) / val_labels_magnitude) * (val_labels_magnitude != 0), nan=0.0)) / torch.count_nonzero(val_labels_magnitude)
                    # 边缘区域的MRE
                    if torch.any(edge_mask_foreground) and torch.any(val_labels_magnitude[edge_mask_foreground] != 0):
                        mre_loss_edge_ = torch.mean(torch.abs((val_outputs_magnitude[edge_mask_foreground] - val_labels_magnitude[edge_mask_foreground]) / val_labels_magnitude[edge_mask_foreground])[(val_labels_magnitude[edge_mask_foreground] != 0)])
                    else:
                        mre_loss_edge_ = torch.tensor(float('nan')).to(device)
                    # MSE: 使用矢量幅值的均方误差
                    mse_loss_ = torch.mean((val_outputs_magnitude - val_labels_magnitude) ** 2)
                    # F-norm: 使用矢量幅值的Frobenius范数（实际上是L2范数）
                    f_norm_ = torch.sqrt(torch.sum((val_outputs_magnitude - val_labels_magnitude) ** 2))
                    # PSNR: 使用矢量幅值的最大值和MSE
                    max_magnitude = torch.max(torch.max(val_outputs_magnitude), torch.max(val_labels_magnitude))
                    psnr_ = 20 * torch.log10(max_magnitude) - 10 * torch.log10(mse_loss_)
                    mae_loss.append(mae_loss_)
                    mae_loss_edge.append(mae_loss_edge_)
                    mae_loss_mean.append(mae_loss_mean_)
                    mae_loss_percentile95.append(mae_loss_percentile95_)
                    mae_loss_percentile99.append(mae_loss_percentile99_)
                    mae_loss_100max.append(mae_loss_100max_)
                    mre_loss.append(mre_loss_)
                    mre_loss_edge.append(mre_loss_edge_)
                    mse_loss.append(mse_loss_)
                    f_norm.append(f_norm_)
                    psnr.append(psnr_)
                    # 计算皮尔逊相关系数（使用矢量幅值）
                    mean_val_labels = torch.mean(val_labels_magnitude)
                    mean_val_outputs = torch.mean(val_outputs_magnitude)

                    # 计算去均值后的张量
                    val_labels_diff = val_labels_magnitude - mean_val_labels
                    val_outputs_diff = val_outputs_magnitude - mean_val_outputs

                    # 计算分子部分（协方差）
                    covariance = torch.sum(val_labels_diff * val_outputs_diff)

                    # 计算分母部分（标准差的乘积）
                    std_val_labels = torch.sqrt(torch.sum(val_labels_diff ** 2))
                    std_val_outputs = torch.sqrt(torch.sum(val_outputs_diff ** 2))
                    denominator = std_val_labels * std_val_outputs

                    # 计算皮尔逊相关系数
                    if denominator != 0:
                        pearson_corr_ = covariance / denominator
                        pearson_corr.append(pearson_corr_)
                    else:
                        pearson_corr.append(torch.tensor(float('nan')).to(device))
                    
                    # 计算矢量夹角（方向差异）
                    # val_outputs_foreground shape: (3, N_foreground), val_labels_foreground shape: (3, N_foreground)
                    # 计算点积
                    dot_product = torch.sum(val_outputs_foreground * val_labels_foreground, dim=0)  # (N_foreground,)
                    # 计算cos(θ) = (v1 · v2) / (|v1| * |v2|)
                    magnitude_product = val_outputs_magnitude * val_labels_magnitude  # (N_foreground,)
                    # 处理幅值为0的情况
                    valid_mask = (magnitude_product != 0)  # (N_foreground,)
                    if torch.any(valid_mask):
                        cos_angle = torch.zeros_like(magnitude_product)
                        cos_angle[valid_mask] = dot_product[valid_mask] / magnitude_product[valid_mask]
                        # 限制在[-1, 1]范围内，防止数值误差
                        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                        # 计算夹角（弧度），然后转换为角度
                        angle_rad = torch.acos(cos_angle[valid_mask])  # (N_valid,)
                        angle_deg = torch.rad2deg(angle_rad)  # 转换为角度
                        # 计算平均夹角（只考虑有效值）
                        vector_angle_ = torch.mean(angle_deg)
                        vector_angle.append(vector_angle_)
                    else:
                        vector_angle.append(torch.tensor(float('nan')).to(device))
                    
                    # 计算矢量差的模长（综合评估方向差异和模长差异）
                    # diff = val_outputs_foreground - val_labels_foreground  # (3, N_foreground)
                    vector_diff = val_outputs_foreground - val_labels_foreground  # (3, N_foreground)
                    # 计算差的模长
                    vector_diff_magnitude_ = torch.sqrt(torch.sum(vector_diff ** 2, dim=0))  # (N_foreground,)
                    # 计算平均差的模长
                    vector_diff_magnitude_mean = torch.mean(vector_diff_magnitude_)
                    vector_diff_magnitude.append(vector_diff_magnitude_mean)
                    
                    evaluation_items=evaluation_per_category(evaluation_items, val_outputs, val_labels, val_images_origin, edge_mask_batch,i)
                pbar.update(1)  # 更新进度条
    mae_metric = torch.mean(torch.tensor(mae_loss))  # 已提前平均到体素上
    mae_edge_metric = torch.mean(torch.tensor(mae_loss_edge)[torch.isfinite(torch.tensor(mae_loss_edge))])  # 已提前平均到体素上
    mre_metric = torch.mean(torch.tensor(mre_loss))  # 已提前平均到体素上
    mre_edge_metric = torch.mean(torch.tensor(mre_loss_edge)[torch.isfinite(torch.tensor(mre_loss_edge))])  # 已提前平均到体素上
    mse_metric = torch.mean(torch.tensor(mse_loss))  # 已提前平均到体素上
    f_norm_metric = torch.mean(torch.tensor(f_norm))
    pearson_corr_metric=torch.mean(torch.tensor(pearson_corr))
    psnr_metric=torch.mean(torch.tensor(psnr))  # 已提前平均到体素上
    mean_metric = torch.mean(torch.tensor(mae_loss_mean))
    percentile95_metric = torch.mean(torch.tensor(mae_loss_percentile95))
    percentile99_metric = torch.mean(torch.tensor(mae_loss_percentile99))
    max100_metric = torch.mean(torch.tensor(mae_loss_100max))
    vector_angle_metric = torch.mean(torch.tensor(vector_angle)[torch.isfinite(torch.tensor(vector_angle))])  # 矢量夹角
    vector_diff_magnitude_metric = torch.mean(torch.tensor(vector_diff_magnitude))  # 矢量差的模长
    output_dir_total=os.path.join(output_dir,'total')
    if not os.path.exists(output_dir_total):
        os.makedirs(output_dir_total)
    with open(os.path.join(output_dir_total,'results_total.txt'), 'w') as f_total:
        print('mae: ', mae_metric, file=f_total)
        print('mre: ', mre_metric, file=f_total)
        print('mse: ', mse_metric, file=f_total)
        print('f norm: ', f_norm_metric, file=f_total)
        print('pearson_corr: ', pearson_corr_metric, file=f_total)
        print('psnr: ', psnr_metric, file=f_total)
        print('mae of edge: ', mae_edge_metric, file=f_total)
        print('mre of edge: ', mre_edge_metric, file=f_total)
        print('mae of mean: ', mean_metric, file=f_total)
        print('mae of percentile 95: ', percentile95_metric, file=f_total)
        print('mae of percentile 99: ', percentile99_metric, file=f_total)
        print('mae of 100max: ', max100_metric, file=f_total)
        print('vector angle (degrees): ', vector_angle_metric, file=f_total)
        print('vector diff magnitude: ', vector_diff_magnitude_metric, file=f_total)
    f_total.close()
    np.save(os.path.join(output_dir_total,'mae_total.npy'),torch.tensor(mae_loss).numpy())
    np.save(os.path.join(output_dir_total, 'mre_total.npy'), torch.tensor(mre_loss).numpy())
    np.save(os.path.join(output_dir_total, 'mse_total.npy'), torch.tensor(mse_loss).numpy())
    np.save(os.path.join(output_dir_total, 'f_norm_total.npy'), torch.tensor(f_norm).numpy())
    np.save(os.path.join(output_dir_total, 'pearson_corr_total.npy'), torch.tensor(pearson_corr).numpy())
    np.save(os.path.join(output_dir_total, 'psnr_total.npy'), torch.tensor(psnr).numpy())
    np.save(os.path.join(output_dir_total, 'mae_loss_edge_total.npy'), torch.tensor(mae_loss_edge)[torch.isfinite(torch.tensor(mae_loss_edge))].numpy())
    np.save(os.path.join(output_dir_total, 'mre_loss_edge_total.npy'), torch.tensor(mre_loss_edge)[torch.isfinite(torch.tensor(mre_loss_edge))].numpy())
    np.save(os.path.join(output_dir_total, 'mae_loss_mean_total.npy'), torch.tensor(mae_loss_mean).numpy())
    np.save(os.path.join(output_dir_total, 'mae_loss_percentile95_total.npy'), torch.tensor(mae_loss_percentile95).numpy())
    np.save(os.path.join(output_dir_total, 'mae_loss_percentile99_total.npy'), torch.tensor(mae_loss_percentile99).numpy())
    np.save(os.path.join(output_dir_total, 'mae_loss_100max_total.npy'), torch.tensor(mae_loss_100max).numpy())
    np.save(os.path.join(output_dir_total, 'vector_angle_total.npy'), torch.tensor(vector_angle)[torch.isfinite(torch.tensor(vector_angle))].numpy())
    np.save(os.path.join(output_dir_total, 'vector_diff_magnitude_total.npy'), torch.tensor(vector_diff_magnitude).numpy())
    for category in category_names:
        output_dir_category=os.path.join(output_dir,'class_'+category)
        if not os.path.exists(output_dir_category):
            os.makedirs(output_dir_category)
        f_category=open(os.path.join(output_dir_category,'results_class_'+category+'.txt'),'w')
        for index_name in index_names:
            np.save(os.path.join(output_dir_category,index_name+'_class_'+category+'.npy'),torch.tensor(evaluation_items[category][index_name]).numpy())
            print(index_name,': ',torch.mean(torch.tensor(evaluation_items[category][index_name])), file=f_category)
        f_category.close()


def append_value(total,item):
    if total is None:
        total=item.clone()
    else:
        total=torch.cat((total,item))
    return total


def edge_mask(data,num_classes,device):
    post_label = AsDiscrete(to_onehot=num_classes)
    edge_data=None
    for data_i in decollate_batch(data):
        data_i = post_label(data_i)
        edge_batch = None
        for j in range(num_classes):  # 包括背景的轮廓
            data_i_j=data_i[j]
            if device=='cpu':
                data_i_j = np.asarray(data_i_j.detach().to(device).numpy(),dtype=bool)
                edge_data_i_j = torch.from_numpy(binary_erosion(data_i_j)^data_i_j)
            else:
                data_i_j = convert_to_cupy(data_i_j,dtype=bool)
                edge_data_i_j = convert_to_tensor(cucim.skimage.morphology.binary_erosion(data_i_j) ^ data_i_j, dtype=bool)
            if edge_batch is None:
                edge_batch=edge_data_i_j[None,:]
            else:
                edge_batch=edge_batch|edge_data_i_j[None,:]
        edge_batch_noborder = torch.zeros_like(edge_batch)  # 由于腐蚀算子的原因在张量的边界处无法区分类别间的边界，因此张量的边界处假设无类别间的边界，否则算子会将张量的边界处认为是类别间的边界
        edge_batch_noborder[:,1:edge_batch_noborder.shape[1]-1,1:edge_batch_noborder.shape[2]-1,1:edge_batch_noborder.shape[3]-1]=edge_batch[:,1:edge_batch_noborder.shape[1]-1,1:edge_batch_noborder.shape[2]-1,1:edge_batch_noborder.shape[3]-1]
        edge_data=append_value(edge_data,edge_batch_noborder[None,:])
    return edge_data


def evaluation_per_category(evaluation_items,val_outputs,val_labels,val_images_origin,edge_mask_batch,batch_i):
    for k in evaluation_items:
        # import pdb
        # pdb.set_trace()
        val_images_category_flag=(val_images_origin[batch_i] == int(k))  # (H, W, D)
        if torch.all(val_images_category_flag==False):
            continue
        # 对于矢量，val_outputs shape: (3, H, W, D), val_labels shape: (3, H, W, D)
        # 提取该类别区域的矢量值：使用masked_select然后reshape
        val_outputs_category = val_outputs[batch_i][:, torch.squeeze(val_images_category_flag,dim=0)]
        val_labels_category = val_labels[batch_i][:, torch.squeeze(val_images_category_flag,dim=0)]
        val_outputs_magnitude_category = torch.sqrt(torch.sum(val_outputs_category ** 2, dim=0))  # (N_category_spatial,)
        val_labels_magnitude_category = torch.sqrt(torch.sum(val_labels_category ** 2, dim=0))  # (N_category_spatial,)
        # 边缘掩码（针对前景区域）
        edge_mask_category = edge_mask_batch[batch_i][val_images_category_flag]  # (N_category_spatial,)
        # MAE: 使用矢量幅值的平均绝对误差
        mae_loss_ = torch.mean(torch.abs(val_outputs_magnitude_category - val_labels_magnitude_category))
        # 边缘区域的MAE（使用矢量幅值）
        if torch.any(edge_mask_category):
            mae_loss_edge_ = torch.mean(torch.abs(val_outputs_magnitude_category[edge_mask_category] - val_labels_magnitude_category[edge_mask_category]))
        # 均值差异（使用矢量幅值）
        mae_loss_mean_ = torch.abs(torch.mean(val_outputs_magnitude_category) - torch.mean(val_labels_magnitude_category))
        # MRE: 使用矢量幅值计算相对误差
        mre_loss_ = torch.sum(torch.nan_to_num(torch.abs((val_outputs_magnitude_category - val_labels_magnitude_category) / val_labels_magnitude_category) * (val_labels_magnitude_category != 0), nan=0.0)) / torch.count_nonzero(val_labels_magnitude_category)
        # 边缘区域的MRE
        if torch.any(edge_mask_category) and torch.any(val_labels_magnitude_category[edge_mask_category] != 0):
            mre_loss_edge_ = torch.mean(torch.abs((val_outputs_magnitude_category[edge_mask_category] - val_labels_magnitude_category[edge_mask_category]) / val_labels_magnitude_category[edge_mask_category])[(val_labels_magnitude_category[edge_mask_category] != 0)])
        # MSE: 使用矢量幅值的均方误差
        mse_loss_ = torch.mean((val_outputs_magnitude_category - val_labels_magnitude_category) ** 2)
        # F-norm: 使用矢量幅值的Frobenius范数（实际上是L2范数）
        f_norm_ = torch.sqrt(torch.sum((val_outputs_magnitude_category - val_labels_magnitude_category) ** 2))
        # PSNR: 使用矢量幅值的最大值和MSE
        max_magnitude = torch.max(torch.max(val_outputs_magnitude_category), torch.max(val_labels_magnitude_category))
        psnr_ = 20 * torch.log10(max_magnitude) - 10 * torch.log10(mse_loss_)
        # 皮尔逊相关系数（使用矢量幅值）
        mean_val_labels = torch.mean(val_labels_magnitude_category)
        mean_val_outputs = torch.mean(val_outputs_magnitude_category)
        val_labels_diff = val_labels_magnitude_category - mean_val_labels
        val_outputs_diff = val_outputs_magnitude_category - mean_val_outputs
        covariance = torch.sum(val_labels_diff * val_outputs_diff)
        std_val_labels = torch.sqrt(torch.sum(val_labels_diff ** 2))
        std_val_outputs = torch.sqrt(torch.sum(val_outputs_diff ** 2))
        denominator = std_val_labels * std_val_outputs
        if denominator != 0:
            pearson_corr_ = covariance / denominator
            if torch.isfinite(pearson_corr_):
                evaluation_items[k]['pearson_corr'].append(pearson_corr_)
        
        # 计算矢量夹角（方向差异）
        # val_outputs_category shape: (3, N_category), val_labels_category shape: (3, N_category)
        # 计算点积
        dot_product = torch.sum(val_outputs_category * val_labels_category, dim=0)  # (N_category,)
        # 计算cos(θ) = (v1 · v2) / (|v1| * |v2|)
        magnitude_product = val_outputs_magnitude_category * val_labels_magnitude_category  # (N_category,)
        # 处理幅值为0的情况
        valid_mask = (magnitude_product != 0)  # (N_category,)
        if torch.any(valid_mask):
            cos_angle = torch.zeros_like(magnitude_product)
            cos_angle[valid_mask] = dot_product[valid_mask] / magnitude_product[valid_mask]
            # 限制在[-1, 1]范围内，防止数值误差
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            # 计算夹角（弧度），然后转换为角度
            angle_rad = torch.acos(cos_angle[valid_mask])  # (N_valid,)
            angle_deg = torch.rad2deg(angle_rad)  # 转换为角度
            # 计算平均夹角（只考虑有效值）
            vector_angle_category = torch.mean(angle_deg)
            if torch.isfinite(vector_angle_category):
                evaluation_items[k]['vector_angle'].append(vector_angle_category)
        
        # 计算矢量差的模长（综合评估方向差异和模长差异）
        vector_diff = val_outputs_category - val_labels_category  # (3, N_category)
        # 计算差的模长
        vector_diff_magnitude_category = torch.sqrt(torch.sum(vector_diff ** 2, dim=0))  # (N_category,)
        # 计算平均差的模长
        vector_diff_magnitude_mean_category = torch.mean(vector_diff_magnitude_category)
        if torch.isfinite(vector_diff_magnitude_mean_category):
            evaluation_items[k]['vector_diff_magnitude'].append(vector_diff_magnitude_mean_category)
        
        if torch.isfinite(mae_loss_):
            evaluation_items[k]['mae'].append(mae_loss_)
        if torch.isfinite(mre_loss_):
            evaluation_items[k]['mre'].append(mre_loss_)
        if torch.isfinite(mse_loss_):
            evaluation_items[k]['mse'].append(mse_loss_)
        if torch.isfinite(f_norm_):
            evaluation_items[k]['f_norm'].append(f_norm_)
        if torch.isfinite(psnr_):
            evaluation_items[k]['psnr'].append(psnr_)
        if torch.isfinite(mae_loss_edge_):
            evaluation_items[k]['mae_loss_edge'].append(mae_loss_edge_)
        if torch.isfinite(mre_loss_edge_):
            evaluation_items[k]['mre_loss_edge'].append(mre_loss_edge_)
        if torch.isfinite(mae_loss_mean_):
            evaluation_items[k]['mae_loss_mean'].append(mae_loss_mean_)
    return evaluation_items


if __name__ == "__main__":
    # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
    # # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/local_seg_final'
    # temp_label_path = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_l.txt'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/31_best_metric_model_efieldcal_array_true_1_interpolate_all.pth'
    # output_dir=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/evaluation/31_based_gtlabel'
    temp_img_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    temp_label_path = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
    split_test_txt = r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/val_l.txt'
    weight_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/weights/42_4_best_metric_model_efieldcal_vector_array_fusion_1_dual_channels.pth'
    output_dir=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/42_4_based_gtlabel_efield_vector'
    dAdt_file=r'/data/disk_2/zhoujunfeng/code/efield_calculation/dadt_808032_0mm_fmm3d.npy'
    main(temp_img_path, temp_label_path, split_test_txt,weight_path,output_dir, dAdt_file)
