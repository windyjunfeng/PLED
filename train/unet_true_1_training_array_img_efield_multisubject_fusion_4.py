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
from mse_loss import MSELoss
from mae_loss import MAELoss
import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    RandRotate90,
    ResizeWithPadOrCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image
from networks.unet_true_1_fusion_4 import UNet_true_1_fusion_4

def main(temp_img_path,temp_label_path,split_train_txt,split_val_txt,dAdt_file):
    num_classes = 1
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # # create a temporary directory and 40 random image, mask pairs
    # print(f"generating synthetic data to {tempdir} (this may take a while)")
    # for i in range(40):
    #     im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)
    #
    #     n = nib.Nifti1Image(im, np.eye(4))
    #     nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))
    #
    #     n = nib.Nifti1Image(seg, np.eye(4))
    #     nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))
    #
    # images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    # segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    f1 = open(split_train_txt, 'r')
    lines_1 = f1.readlines()
    train_imgs=[]
    train_labels=[]
    for line in lines_1:
        temp_img_path_whole=os.path.join(temp_img_path, line.rstrip('\n'), 'local_sampling_scalp_labels_new/local_label')
        temp_label_path_whole=os.path.join(temp_label_path, line.rstrip('\n'), 'local_efield')
        files=os.listdir(temp_img_path_whole)
        for file in files:
            train_imgs.append(os.path.join(temp_img_path_whole, file))
            train_labels.append(os.path.join(temp_label_path_whole, file))
    f2 = open(split_val_txt, 'r')
    lines_2 = f2.readlines()
    val_imgs=[]
    val_labels=[]
    for line in lines_2:
        temp_img_path_whole=os.path.join(temp_img_path, line.rstrip('\n'), 'local_sampling_scalp_labels_new/local_label')
        temp_label_path_whole=os.path.join(temp_label_path, line.rstrip('\n'), 'local_efield')
        files=os.listdir(temp_img_path_whole)
        for file in files:
            val_imgs.append(os.path.join(temp_img_path_whole, file))
            val_labels.append(os.path.join(temp_label_path_whole, file))
    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            # ScaleIntensity(),
            EnsureChannelFirst(),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            # ResizeWithPadOrCrop(spatial_size=[80, 80, 32])
        ]
    )
    train_segtrans = Compose(
        [
            EnsureChannelFirst(),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            # ResizeWithPadOrCrop(spatial_size=[80, 80, 32])
        ]
    )
    # val_imtrans = Compose([ScaleIntensity(), EnsureChannelFirst(), ResizeWithPadOrCrop(spatial_size=[80, 80, 32])])
    # val_segtrans = Compose([EnsureChannelFirst(), ResizeWithPadOrCrop(spatial_size=[80, 80, 32])])
    val_imtrans = Compose([EnsureChannelFirst(),])
    val_segtrans = Compose([EnsureChannelFirst(),])

    # # define image dataset, data loader
    # check_ds = ImageDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
    # check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, seg = monai.utils.misc.first(check_loader)
    # print(im.shape, seg.shape)
    # import pdb
    # pdb.set_trace()
    # create a training data loader
    train_ds = ImageDataset(train_imgs, train_labels, transform=train_imtrans, seg_transform=train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = ImageDataset(val_imgs, val_labels, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=8, pin_memory=torch.cuda.is_available())


    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_true_1_fusion_4(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(8, 16, 32, 64, 128),
        strides=(1, 2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    # loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    loss_function=MSELoss()
    # loss_function=MAELoss()

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    total_epoch = 30
    val_interval = 1
    best_metric = 1000000
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    # import pdb
    # pdb.set_trace()
    dAdt=np.load(dAdt_file)
    dAdt = torch.tensor(dAdt).to(device)
    conductivity_map = {'1':0.126,'2':0.275,'3':1.654,'4':0.01,'5':0.465,'9':0.6}
    for epoch in range(total_epoch):
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{total_epoch}", unit='batch') as pbar:
            for batch_data in train_loader:
                step += 1
                # import pdb
                # pdb.set_trace()
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                for k in conductivity_map:
                    inputs=torch.where(inputs==int(k),torch.tensor(conductivity_map[k]).to(torch.float32).to(device),inputs)
                dAdt_train = dAdt.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1, 1)
                dAdt_mask=torch.where(inputs==0,torch.tensor(0).to(torch.float32).to(device),dAdt_train)  # 利用分割结果作掩码
                # cat_inputs = torch.cat((inputs,dAdt_mask),1)
                optimizer.zero_grad()
                # outputs = model(cat_inputs)
                # import pdb
                # pdb.set_trace()
                outputs=model(inputs,dAdt_mask)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})  # set_postfix接受一个字典对象
                pbar.update(1)  # 更新进度条
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                mse_loss=None
                pearson_corr = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    for k in conductivity_map:
                        val_images = torch.where(val_images == int(k), torch.tensor(conductivity_map[k]).to(torch.float32).to(device), val_images)
                    dAdt_val = dAdt.unsqueeze(0).repeat(val_images.shape[0], 1, 1, 1, 1)  # 这里不能是batch size，因为有些测试集的图片数不能被batch size整除
                    dAdt_mask = torch.where(val_images == 0, torch.tensor(0).to(torch.float32).to(device), dAdt_val)  # 利用分割结果作掩码
                    # cat_inputs = torch.cat((val_images, dAdt_mask), 1)
                    # val_outputs = model(cat_inputs)
                    val_outputs = model(val_images, dAdt_mask)
                    # compute metric for current iteration
                    # mse_loss_ = torch.norm(val_outputs - val_labels , p='fro')
                    mse_loss_ = torch.sqrt(torch.sum((val_outputs - val_labels) ** 2, dim=(1,2,3,4)))
                    if mse_loss is None:
                        # mse_loss=mse_loss_.unsqueeze(0)
                        mse_loss = mse_loss_.clone()
                    else:
                        # mse_loss=torch.cat((mse_loss, mse_loss_.unsqueeze(0)))
                        mse_loss = torch.cat((mse_loss, mse_loss_))
                    # 计算均值
                    mean_val_labels = torch.mean(val_labels, dim=(1,2,3,4), keepdim=True)
                    mean_val_outputs = torch.mean(val_outputs, dim=(1,2,3,4), keepdim=True)
                    # 计算去均值后的张量
                    val_labels_diff = val_labels - mean_val_labels
                    val_outputs_diff = val_outputs - mean_val_outputs
                    # 计算分子部分（协方差）
                    covariance = torch.sum(val_labels_diff * val_outputs_diff, dim=(1,2,3,4))
                    # 计算分母部分（标准差的乘积）
                    std_val_labels = torch.sqrt(torch.sum(val_labels_diff ** 2, dim=(1,2,3,4)))
                    std_val_outputs = torch.sqrt(torch.sum(val_outputs_diff ** 2, dim=(1,2,3,4)))
                    denominator = std_val_labels * std_val_outputs
                    # 计算皮尔逊相关系数
                    pearson_corr_ = torch.where(denominator != 0, covariance / denominator, torch.zeros_like(denominator))  # 每个皮尔逊相关系数都是一张图的
                    if pearson_corr is None:
                        # pearson_corr = pearson_corr_.unsqueeze(0)
                        pearson_corr = pearson_corr_.clone()  # 注意不能直接等于，否则两者会相互影响
                    else:
                        # pearson_corr = torch.cat((pearson_corr, pearson_corr_.unsqueeze(0)))
                        pearson_corr = torch.cat((pearson_corr, pearson_corr_))
                # aggregate the final mean dice result
                metric = torch.mean(mse_loss)
                # import pdb
                # pdb.set_trace()
                # metric = metric/(val_outputs.shape[0]*val_outputs.shape[1])
                pearson_corr_metric = torch.mean(pearson_corr)
                # reset the status for next validation round
                metric_values.append(metric)
                if metric < best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_efieldcal_array_fusion_4.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean mse loss: {:.4f} best mean mse loss: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                print("pearson correlation coefficient: {:.4f}".format(pearson_corr_metric))
                writer.add_scalar("val_mse_loss", metric, epoch + 1)
                writer.add_scalar("val_pearson_correlation_coefficient", pearson_corr_metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    temp_img_path=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
    temp_label_path=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/local_efield'
    split_train_txt=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/train.txt'
    split_val_txt=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/val.txt'
    dAdt_file=r'/root/autodl-tmp/tms_e-field_scalp_data/efield_calculation/dadt_norm_808032_0mm_fmm3d.npy'
    main(temp_img_path,temp_label_path,split_train_txt,split_val_txt,dAdt_file)
