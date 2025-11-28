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
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,HausdorffDistanceMetric,MeanIoU
# from monai.networks.nets import UNet
from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, SaveImage, ScaleIntensity,ResizeWithPadOrCrop
from networks.unet_true_1 import UNet_true_1

def main(temp_img_path,temp_label_path,temp_lesion_path, split_test_txt,weight_path,output_dir):
    num_classes = 10
    num_classes_add_lesion = 20
    true_class = [0,1,2,3,4,5,9,11,12,13,14,15,19]
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    f1 = open(split_test_txt, 'r')
    lines_1 = f1.readlines()
    test_imgs=[]
    test_labels=[]
    # define transforms for image and segmentation
    for line in lines_1:
        temp_img_path_whole = os.path.join(temp_img_path, line.rstrip('\n'),
                                           'local_sampling_scalp_imgs_new/local_image')
        temp_label_path_whole = os.path.join(temp_label_path, line.rstrip('\n'),
                                             'local_sampling_scalp_labels_new/local_label')
        files = os.listdir(temp_img_path_whole)
        for file in files:
            test_imgs.append(os.path.join(temp_img_path_whole, file))
            test_labels.append(os.path.join(temp_label_path_whole, file))
    imtrans = Compose([ScaleIntensity(),EnsureChannelFirst(),])
    # imtrans = Compose([EnsureChannelFirst(), ])
    segtrans = Compose([EnsureChannelFirst(),])
    val_ds = ImageDataset(test_imgs, test_labels, transform=imtrans, seg_transform=segtrans, image_only=False)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=8, pin_memory=torch.cuda.is_available())
    dice_metric_class = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, num_classes=num_classes_add_lesion)
    # dice_metric_class = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, num_classes=num_classes)
    haus_metric_class = HausdorffDistanceMetric(include_background=True, reduction='mean')
    # haus_metric_class = HausdorffDistanceMetric(include_background=True, reduction='mean_batch')
    iou_metric_class = MeanIoU(include_background=True, reduction='mean')
    # iou_metric_class = MeanIoU(include_background=True, reduction='mean_batch')
    post_label = AsDiscrete(to_onehot=num_classes_add_lesion)
    post_pred_1 = AsDiscrete(argmax=True)  # 先取值最大的通道索引，然后one_hot编码
    post_pred_2 = AsDiscrete(to_onehot=num_classes_add_lesion)
    # saver = SaveImage(output_dir="./output", output_ext=".nii.gz", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=num_classes,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(device)
    index_names = ['dice', 'hausdorff_distance', 'iou']
    # evaluation_items = {category: {index_name: [] for index_name in index_names} for category in category_names}
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
    with tqdm(total=len(val_loader)) as pbar:
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                val_outputs_origin = model(val_images)
                val_outputs = []
                # import pdb
                # pdb.set_trace()
                for j in range(val_images.shape[0]):
                    val_output = post_pred_1(val_outputs_origin[j])
                    val_lesion_path=os.path.join(val_output.meta['filename_or_obj'].split('/local_sampling_scalp_imgs_new/local_image/')[0],'local_lesion',os.path.basename(val_output.meta['filename_or_obj']))
                    if os.path.exists(val_lesion_path):
                        # import pdb
                        # pdb.set_trace()
                        val_lesion=torch.tensor(np.load(val_lesion_path))[None, :].to(device)
                        val_output = val_output + val_lesion * 10
                    val_output = post_pred_2(val_output)
                    val_outputs.append(val_output)
                # compute metric for current iteration
                # import pdb
                # pdb.set_trace()
                dice_metric_class(y_pred=val_outputs, y=val_labels)
                haus_metric_class(y_pred=val_outputs, y=val_labels)
                iou_metric_class(y_pred=val_outputs, y=val_labels)
                # for val_output in val_outputs:
                #     saver(val_output)  # 保存预测结果
                pbar.update(1)  # 更新进度条
    output_dir_total = os.path.join(output_dir, 'total')
    if not os.path.exists(output_dir_total):
        os.makedirs(output_dir_total)
    # import pdb
    # pdb.set_trace()
    inf_mask=torch.isinf(haus_metric_class.get_buffer())
    temp=haus_metric_class.get_buffer().clone()
    temp[inf_mask]=float('nan')  # hausdorff distance中除了nan还有inf，但monai中不支持对inf的置换
    haus_metric_class.reset()
    haus_metric_class.append(temp[0])
    haus_metric_class.extend(temp[1:])
    with open(os.path.join(output_dir_total,'results_total.txt'), 'w') as f:
        print("dice for each class:", dice_metric_class.aggregate(reduction='mean_batch')[true_class], file=f)
        print("average dice:", dice_metric_class.aggregate(reduction='mean'),file=f)
        print('\n')
        print("hausdorff distance for each class:", haus_metric_class.aggregate(reduction='mean_batch')[true_class],file=f)
        print("average hausdorff distance:", haus_metric_class.aggregate(reduction='mean'),file=f)
        print('\n')
        print("iou for each class:", iou_metric_class.aggregate(reduction='mean_batch')[true_class],file=f)
        print("average iou:", iou_metric_class.aggregate(reduction='mean'),file=f)
    f.close()
    np.save(os.path.join(output_dir_total, 'dice_total.npy'),dice_metric_class.aggregate(reduction='mean_channel').cpu().numpy())
    np.save(os.path.join(output_dir_total, 'hausdorff_distance_total.npy'),haus_metric_class.aggregate(reduction='mean_channel').cpu().numpy())
    np.save(os.path.join(output_dir_total, 'iou_total.npy'),iou_metric_class.aggregate(reduction='mean_channel').cpu().numpy())
    for category in true_class:
        output_dir_category = os.path.join(output_dir, 'class_' + str(category))
        if not os.path.exists(output_dir_category):
            os.makedirs(output_dir_category)
        for index_name in index_names:
            if index_name=='dice':
                np.save(os.path.join(output_dir_category, index_name + '_class_' + str(category) + '.npy'), dice_metric_class.get_buffer()[:,category].cpu().numpy())
            elif index_name=='hausdorff_distance':
                np.save(os.path.join(output_dir_category, index_name + '_class_' + str(category) + '.npy'), haus_metric_class.get_buffer()[:, category].cpu().numpy())
            elif index_name=='iou':
                np.save(os.path.join(output_dir_category, index_name + '_class_' + str(category) + '.npy'), iou_metric_class.get_buffer()[:, category].cpu().numpy())


if __name__ == "__main__":
    temp_img_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    temp_label_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    temp_lesion_rootpath = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/m2m_files_add_lesion'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_l.txt'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/32_best_best_metric_model_segmentation3d_array_scalp_true_1_all.pth'
    # output_dir=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/evaluation/32_final'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/val_s.txt'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/29_best_metric_model_segmentation3d_array_scalp_true_1_generaldiceloss.pth'
    # output_dir=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/evaluation/29_s'
    # split_test_txt = r'/root/autodl-tmp/tms_e-field_scalp_data/cog_test.txt'
    # weight_path=r'/root/autodl-fs/code/deep_learning_efield/segmentation/unet/32_best_best_metric_model_segmentation3d_array_scalp_true_1_all.pth'
    # output_dir=r'/root/autodl-tmp/tms_e-field_scalp_data/segmentation/evaluation/32_cog'
    split_test_txt = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/test.txt'
    weight_path = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/code/deep_learning_e-field/segmentation/unet/32_best_best_metric_model_segmentation3d_array_scalp_true_1_all.pth'
    output_dir = r'/media/irc300/d37c3a3a-b751-4342-9504-f7ee31232551/windyjunfeng/data/deep_learning_e-field/TMS-EEG_lesion_cortex/evaluation/segmentation/32_final'
    main(temp_img_path, temp_label_path, temp_lesion_rootpath, split_test_txt,weight_path,output_dir)
