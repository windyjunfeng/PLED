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
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, SaveImage, ScaleIntensity,ResizeWithPadOrCrop
from networks.unet_true_1 import UNet_true_1
from monai.utils import convert_to_cupy,convert_to_tensor


# mp.set_sharing_strategy('file_system')


def main(temp_img_path,temp_label_path,split_test_txt,weight_path, output_dir):
    num_classes = 1
    # num_labels=10
    num_labels = 20
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
        # temp_img_path_whole=os.path.join(temp_img_path, line.rstrip('\n'), 'local_sampling_scalp_labels_new/local_label')
        # temp_img_path_whole = os.path.join(temp_img_path, line.rstrip('\n'))
        temp_img_path_whole = os.path.join(temp_img_path, line.rstrip('\n'), 'inference/local_seg_final')
        # temp_label_path_whole=os.path.join(temp_label_path, line.rstrip('\n'), 'local_efield')
        temp_label_path_whole = os.path.join(temp_label_path, line.rstrip('\n'),'mesh2nii_index_new_correct_interpolate/efield_calculation/local_efield')
        files=os.listdir(temp_img_path_whole)
        for file in files:
            test_imgs.append(os.path.join(temp_img_path_whole, file))
            test_labels.append(os.path.join(temp_label_path_whole, file))
    # define transforms for image and segmentation
    imtrans = Compose([EnsureChannelFirst()])
    segtrans = Compose([EnsureChannelFirst()])
    val_ds = ImageDataset(test_imgs, test_labels, transform=imtrans, seg_transform=segtrans)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())  # 将一个batch的数据拆开时这样会报错
    # val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=8, pin_memory=False)
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
    # category_names = ['1','2','3']
    # category_names = ['4','5','9']
    category_names = ['9', '11', '12', '13', '14', '15', '19']
    index_names = ['mae','mre','mse','f_norm','pearson_corr','psnr','mae_loss_edge','mre_loss_edge','mae_loss_mean']
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
            evaluation_items = {category: {index_name: [] for index_name in index_names} for category in category_names}
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                # import pdb
                # pdb.set_trace()
                val_images_origin = val_images.clone()
                val_outputs = model(val_images)
                edge_mask_batch = edge_mask(val_images_origin, num_labels, device).to(device)
                for i in range(val_images.shape[0]):  # 这里不应该是batch size，因为最后一个batch不一定是满batch的
                    # val_outputs_foreground=val_outputs[i][val_images[i] != 0]
                    # val_labels_foreground=val_labels[i][val_images[i] != 0]
                    # mae_loss_ = torch.mean(torch.abs(val_outputs_foreground - val_labels_foreground))
                    # mae_loss_edge_ = torch.mean(torch.abs((val_outputs_foreground - val_labels_foreground))[edge_mask_batch[i][val_images[i] != 0]])
                    # mae_loss_mean_ = torch.abs(torch.mean(val_outputs_foreground)-torch.mean(val_labels_foreground))
                    # mae_loss_percentile95_=torch.abs(torch.quantile(val_outputs_foreground,0.95)-torch.quantile(val_labels_foreground,0.95))
                    # mae_loss_percentile99_ = torch.abs(torch.quantile(val_outputs_foreground, 0.99) - torch.quantile(val_labels_foreground, 0.99))
                    # mae_loss_100max_ = torch.abs(torch.topk(val_outputs_foreground,100)[0][-1]-torch.topk(val_labels_foreground,100)[0][-1])
                    # mre_loss_ = torch.sum(torch.nan_to_num(torch.abs((val_outputs_foreground - val_labels_foreground)/val_labels_foreground)*(val_labels_foreground!=0),nan=0.0))/torch.count_nonzero(val_labels_foreground)
                    # mre_loss_edge_ = torch.mean(torch.abs((val_outputs_foreground - val_labels_foreground) / val_labels_foreground)[(val_labels_foreground != 0) & edge_mask_batch[i][val_images[i] != 0]])
                    # mse_loss_ = torch.mean((val_outputs_foreground - val_labels_foreground) ** 2)
                    # f_norm_ = torch.sqrt(torch.sum((val_outputs_foreground - val_labels_foreground) ** 2))
                    # psnr_ = 20*torch.log10(torch.max(torch.amax(val_outputs_foreground),torch.amax(val_labels_foreground)))-10*torch.log10(mse_loss_)
                    # mae_loss.append(mae_loss_)
                    # mae_loss_edge.append(mae_loss_edge_)
                    # mae_loss_mean.append(mae_loss_mean_)
                    # mae_loss_percentile95.append(mae_loss_percentile95_)
                    # mae_loss_percentile99.append(mae_loss_percentile99_)
                    # mae_loss_100max.append(mae_loss_100max_)
                    # mre_loss.append(mre_loss_)
                    # mre_loss_edge.append(mre_loss_edge_)
                    # mse_loss.append(mse_loss_)
                    # f_norm.append(f_norm_)
                    # psnr.append(psnr_)
                    # # 计算均值
                    # mean_val_labels = torch.mean(val_labels_foreground)
                    # mean_val_outputs = torch.mean(val_outputs_foreground)
                    #
                    # # 计算去均值后的张量
                    # val_labels_diff = val_labels_foreground - mean_val_labels
                    # val_outputs_diff = val_outputs_foreground - mean_val_outputs
                    #
                    # # 计算分子部分（协方差）
                    # covariance = torch.sum(val_labels_diff * val_outputs_diff)
                    #
                    # # 计算分母部分（标准差的乘积）
                    # std_val_labels = torch.sqrt(torch.sum(val_labels_diff ** 2))
                    # std_val_outputs = torch.sqrt(torch.sum(val_outputs_diff ** 2))
                    # denominator = std_val_labels * std_val_outputs
                    #
                    # # 计算皮尔逊相关系数
                    # # pearson_corr_ = torch.where(denominator != 0, covariance / denominator, torch.zeros_like(denominator))
                    # # if denominator == 0:
                    # #     print('pearson correlation is none: ',val_images[i].meta['filename_or_obj'])
                    # if denominator != 0:
                    #     pearson_corr_ = covariance / denominator
                    #     pearson_corr.append(pearson_corr_)
                    evaluation_items=evaluation_per_category(evaluation_items, val_outputs, val_labels, val_images_origin, edge_mask_batch,i)
                pbar.update(1)  # 更新进度条
    # mae_metric = torch.mean(torch.tensor(mae_loss))  # 已提前平均到体素上
    # mae_edge_metric = torch.mean(torch.tensor(mae_loss_edge)[torch.isfinite(torch.tensor(mae_loss_edge))])  # 已提前平均到体素上
    # mre_metric = torch.mean(torch.tensor(mre_loss))  # 已提前平均到体素上
    # mre_edge_metric = torch.mean(torch.tensor(mre_loss_edge)[torch.isfinite(torch.tensor(mre_loss_edge))])  # 已提前平均到体素上
    # mse_metric = torch.mean(torch.tensor(mse_loss))  # 已提前平均到体素上
    # f_norm_metric = torch.mean(torch.tensor(f_norm))
    # pearson_corr_metric=torch.mean(torch.tensor(pearson_corr))
    # psnr_metric=torch.mean(torch.tensor(psnr))  # 已提前平均到体素上
    # mean_metric = torch.mean(torch.tensor(mae_loss_mean))
    # percentile95_metric = torch.mean(torch.tensor(mae_loss_percentile95))
    # percentile99_metric = torch.mean(torch.tensor(mae_loss_percentile99))
    # max100_metric = torch.mean(torch.tensor(mae_loss_100max))
    # output_dir_total=os.path.join(output_dir,'total')
    # if not os.path.exists(output_dir_total):
    #     os.makedirs(output_dir_total)
    # with open(os.path.join(output_dir_total,'results_total.txt'), 'w') as f_total:
    #     print('mae: ', mae_metric, file=f_total)
    #     print('mre: ', mre_metric, file=f_total)
    #     print('mse: ', mse_metric, file=f_total)
    #     print('f norm: ', f_norm_metric, file=f_total)
    #     print('pearson_corr: ', pearson_corr_metric, file=f_total)
    #     print('psnr: ', psnr_metric, file=f_total)
    #     print('mae of edge: ', mae_edge_metric, file=f_total)
    #     print('mre of edge: ', mre_edge_metric, file=f_total)
    #     print('mae of mean: ', mean_metric, file=f_total)
    #     print('mae of percentile 95: ', percentile95_metric, file=f_total)
    #     print('mae of percentile 99: ', percentile99_metric, file=f_total)
    #     print('mae of 100max: ', max100_metric, file=f_total)
    # f_total.close()
    # np.save(os.path.join(output_dir_total,'mae_total.npy'),torch.tensor(mae_loss).numpy())
    # np.save(os.path.join(output_dir_total, 'mre_total.npy'), torch.tensor(mre_loss).numpy())
    # np.save(os.path.join(output_dir_total, 'mse_total.npy'), torch.tensor(mse_loss).numpy())
    # np.save(os.path.join(output_dir_total, 'f_norm_total.npy'), torch.tensor(f_norm).numpy())
    # np.save(os.path.join(output_dir_total, 'pearson_corr_total.npy'), torch.tensor(pearson_corr).numpy())
    # np.save(os.path.join(output_dir_total, 'psnr_total.npy'), torch.tensor(psnr).numpy())
    # np.save(os.path.join(output_dir_total, 'mae_loss_edge_total.npy'), torch.tensor(mae_loss_edge)[torch.isfinite(torch.tensor(mae_loss_edge))].numpy())
    # np.save(os.path.join(output_dir_total, 'mre_loss_edge_total.npy'), torch.tensor(mre_loss_edge)[torch.isfinite(torch.tensor(mre_loss_edge))].numpy())
    # np.save(os.path.join(output_dir_total, 'mae_loss_mean_total.npy'), torch.tensor(mae_loss_mean).numpy())
    # np.save(os.path.join(output_dir_total, 'mae_loss_percentile95_total.npy'), torch.tensor(mae_loss_percentile95).numpy())
    # np.save(os.path.join(output_dir_total, 'mae_loss_percentile99_total.npy'), torch.tensor(mae_loss_percentile99).numpy())
    # np.save(os.path.join(output_dir_total, 'mae_loss_100max_total.npy'), torch.tensor(mae_loss_100max).numpy())
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
        val_images_category_flag=(val_images_origin[batch_i] == int(k))
        if torch.all(val_images_category_flag==False):
            continue
        val_outputs_foreground = val_outputs[batch_i][val_images_category_flag]
        val_labels_foreground = val_labels[batch_i][val_images_category_flag]
        mae_loss_ = torch.mean(torch.abs(val_outputs_foreground - val_labels_foreground))
        mae_loss_edge_ = torch.mean(torch.abs((val_outputs_foreground - val_labels_foreground))[edge_mask_batch[batch_i][val_images_category_flag]])
        mae_loss_mean_ = torch.abs(torch.mean(val_outputs_foreground) - torch.mean(val_labels_foreground))
        mre_loss_ = torch.sum(torch.nan_to_num(torch.abs((val_outputs_foreground - val_labels_foreground) / val_labels_foreground) * (val_labels_foreground != 0), nan=0.0)) / torch.count_nonzero(val_labels_foreground)
        mre_loss_edge_ = torch.mean(torch.abs((val_outputs_foreground - val_labels_foreground) / val_labels_foreground)[(val_labels_foreground != 0) & edge_mask_batch[batch_i][val_images_category_flag]])
        mse_loss_ = torch.mean((val_outputs_foreground - val_labels_foreground) ** 2)
        f_norm_ = torch.sqrt(torch.sum((val_outputs_foreground - val_labels_foreground) ** 2))
        psnr_ = 20*torch.log10(torch.max(torch.amax(val_outputs_foreground),torch.amax(val_labels_foreground)))-10*torch.log10(mse_loss_)
        mean_val_labels = torch.mean(val_labels_foreground)
        mean_val_outputs = torch.mean(val_outputs_foreground)
        val_labels_diff = val_labels_foreground - mean_val_labels
        val_outputs_diff = val_outputs_foreground - mean_val_outputs
        covariance = torch.sum(val_labels_diff * val_outputs_diff)
        std_val_labels = torch.sqrt(torch.sum(val_labels_diff ** 2))
        std_val_outputs = torch.sqrt(torch.sum(val_outputs_diff ** 2))
        denominator = std_val_labels * std_val_outputs
        # pearson_corr_ = torch.where(denominator != 0, covariance / denominator, torch.zeros_like(denominator))
        # if denominator == 0: # class 9经常出现这种情况
        #     print('pearson correlation is none: ', val_images_origin[batch_i].meta['filename_or_obj'],'class_'+str(k))
        if denominator != 0:
            pearson_corr_ = covariance / denominator
            if torch.isfinite(pearson_corr_):
                evaluation_items[k]['pearson_corr'].append(pearson_corr_)
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
    # temp_img_path = r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/inference/local_seg_final'
    # temp_label_path = r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_l.txt'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/31_best_metric_model_efieldcal_array_true_1_interpolate_all.pth'
    # output_dir=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/evaluation/31_based_step1'
    temp_img_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    temp_label_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    split_test_txt = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/test.txt'
    weight_path=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/code/deep_learning_e-field/segmentation/unet/31_best_metric_model_efieldcal_array_true_1_interpolate_all.pth'
    output_dir=r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/evaluation/efield_calculation/31_based_step1_add_lesion'
    main(temp_img_path, temp_label_path, split_test_txt,weight_path,output_dir)
