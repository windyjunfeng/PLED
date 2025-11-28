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
from networks.unet_true_1 import UNet_true_1

def main(temp_img_path,temp_label_path,split_train_txt,split_val_txt):
    num_classes = 10
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
        temp_img_path_whole=os.path.join(temp_img_path, line.rstrip('\n'), 'local_sampling_scalp_imgs_new/local_image')
        temp_label_path_whole=os.path.join(temp_label_path, line.rstrip('\n'), 'local_sampling_scalp_labels_new/local_label')
        files=os.listdir(temp_img_path_whole)
        for file in files:
            train_imgs.append(os.path.join(temp_img_path_whole, file))
            train_labels.append(os.path.join(temp_label_path_whole, file))
    f2 = open(split_val_txt, 'r')
    lines_2 = f2.readlines()
    val_imgs=[]
    val_labels=[]
    for line in lines_2:
        temp_img_path_whole=os.path.join(temp_img_path, line.rstrip('\n'), 'local_sampling_scalp_imgs_new/local_image')
        temp_label_path_whole=os.path.join(temp_label_path, line.rstrip('\n'), 'local_sampling_scalp_labels_new/local_label')
        files=os.listdir(temp_img_path_whole)
        for file in files:
            val_imgs.append(os.path.join(temp_img_path_whole, file))
            val_labels.append(os.path.join(temp_label_path_whole, file))
    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    train_segtrans = Compose(
        [
            EnsureChannelFirst(),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
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
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = ImageDataset(val_imgs, val_labels, transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=16, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)  # 先取值最大的通道索引，然后one_hot编码

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_true_1(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(8, 16, 32, 64, 128),  # 注意，区别于monai的unet，需要修改
        strides=(1, 2, 2, 2, 2),  # 注意，区别于monai的unet，需要修改
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceFocalLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    total_epoch = 50
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    # import pdb
    # pdb.set_trace()
    for epoch in range(total_epoch):
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{total_epoch}", unit='batch') as pbar:
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
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
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation3d_array_scalp_true_1_dicefocalloss.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    temp_img_path=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/image'
    temp_label_path=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/label'
    split_train_txt=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/train.txt'
    split_val_txt=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/val.txt'
    main(temp_img_path,temp_label_path,split_train_txt,split_val_txt)
