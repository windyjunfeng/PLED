# coding=utf-8
# 考虑了相同最大值和最大topk值相同的情况
import os
import numpy as np
from scipy.spatial import cKDTree


subjects=['m2m_A2','m2m_A3','m2m_A8','m2m_A9','m2m_A14','m2m_A17','m2m_A18','m2m_A20','m2m_A24']
# subjects=['m2m_A2','m2m_A9'] # 仅用于35 epoch
# subjects=['m2m_390645','m2m_432332','m2m_566454','m2m_644044','m2m_753150']
# subjects=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
# subjects=['m2m_TMS-015','m2m_TMS-040','m2m_TMS-076','m2m_TMS-103']
categories=['gm','foreground']
# categories=['gm']
gt_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/gt_optimization_vector/optimal_eval'
sampling_points_root_path=r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health'
mean_efield_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/46_based_gtlabel_optimization_vector/mean_efield'
output_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/46_based_gtlabel_optimization_vector/optimal_eval'
# mean_efield_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\optimization_evaluation\35_cog\mean_efield'
# output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\optimization_evaluation\35_cog\optimal_eval'
# mean_efield_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\optimization_evaluation\31_based_step1_cog\mean_efield'
# output_root_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\optimization_evaluation\31_based_step1_cog\optimal_eval'
num_neighbors_pre=25  # 预选的邻近点数
radius_search=10
angle_num=12
angle_dict={'0':-90,'1':-75,'2':-60,'3':-45,'4':-30,'5':-15,'6':0,'7':15,'8':30,'9':45,'10':60,'11':75}
# angle_dict={str(i+90):i for i in range(-90,90,1)}   # 180个角度时使用
# angle_num=180


# total_num=0
# best_id_match=0
# best_id_match_correction=0
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)
for subject in subjects:
    # sampling_points_sub_path=os.path.join(sampling_points_root_path,subject,'coil_positions.npy')
    sampling_points_sub_path=os.path.join(sampling_points_root_path,subject,'local_sampling_scalp_labels_new_1010_center/coil_positions.npy')
    sampling_points=np.load(sampling_points_sub_path)
    # import pdb
    # pdb.set_trace()
    for k in categories:
        mean_efield_path=os.path.join(mean_efield_root_path,subject.split('_')[-1]+'_'+k+'_mean_efield.npy')
        mean_efield=np.load(mean_efield_path)

        best_ids_gt=np.load(os.path.join(gt_root_path,subject.split('_')[-1]+'_'+k+'_best_id.npy'))
        best_ids_top3_gt = np.load(os.path.join(gt_root_path, subject.split('_')[-1] + '_' + k + '_best_id_top3.npy'))
        best_ids_top5_gt = np.load(os.path.join(gt_root_path, subject.split('_')[-1] + '_' + k + '_best_id_top5.npy'))

        best_id_total=[]
        best_id_top3_total = []
        best_id_top5_total = []
        best_efield_total=[]
        best_efield_top3_total = []
        best_efield_top5_total = []
        best_pos_total=[]
        best_angle_total=[]
        for i in range(len(sampling_points)):
            point_target_sub=sampling_points[i]
            tree=cKDTree(sampling_points)
            _, neighbors_id = tree.query(point_target_sub, k=num_neighbors_pre)
            sampling_points_neighbors=sampling_points[neighbors_id]
            distances = np.linalg.norm(np.array(point_target_sub) - sampling_points_neighbors, axis=1)
            neighbors_id_keep=neighbors_id[distances<=radius_search]
            total_id=[]
            for n_k in neighbors_id_keep:
                total_id=np.concatenate((total_id,np.arange(n_k*angle_num,(n_k+1)*angle_num))).astype(int)
            best_id = total_id[np.nanargmax(mean_efield[total_id])]  # 排除里面为nan的元素
            # best_id = total_id[np.argmax(mean_efield[total_id])]
            # best_id_top3 = total_id[np.argpartition(mean_efield[total_id],-3)[-3:]]  # 该种方式求解的并不是全局中的topk，只是局部的topk
            # best_id_top5 = total_id[np.argpartition(mean_efield[total_id], -5)[-5:]]
            # import pdb
            # pdb.set_trace()

            nan_mask = np.isnan(mean_efield)
            mean_efield = np.where(nan_mask, -np.inf, mean_efield)  # 将里面的nan替换成-inf，这样取前topk时就不会被取到

            best_id_top3 = total_id[np.argsort(mean_efield[total_id])[-3:][::-1]]
            best_id_top5 = total_id[np.argsort(mean_efield[total_id])[-5:][::-1]]

            best_ids = total_id[np.where(np.isin(mean_efield[total_id],mean_efield[best_id]))[0]]  # 存在相同的最大值
            # total_num=total_num+1
            # if best_ids_gt[i]==best_id:
            #     best_id_match=best_id_match+1
            #     best_id_match_correction=best_id_match_correction+1
            if len(best_ids)>1 and best_ids_gt[i]!=best_id:
                match_flag = False
                if best_ids_gt[i] in best_ids:
                    best_id=best_ids_gt[i]
                    match_flag = True
                    # best_id_match_correction = best_id_match_correction + 1
                elif best_id not in best_ids_top3_gt[i]:
                    for best_id_top3_gt in best_ids_top3_gt[i]:
                        if best_id_top3_gt in best_ids:
                            best_id=best_id_top3_gt
                            match_flag=True
                            break
                    if not match_flag and best_id not in best_ids_top5_gt[i]:
                        for best_id_top5_gt in best_ids_top5_gt[i]:
                            if best_id_top5_gt in best_ids:
                                best_id = best_id_top5_gt
                                match_flag=True
                                break
            best_efield = mean_efield[best_id]
            best_efield_top3 = mean_efield[best_id_top3]
            best_efield_top5 = mean_efield[best_id_top5]
            best_pos=sampling_points[best_id//angle_num]
            best_angle=angle_dict[str(best_id%angle_num)]
            best_id_total.append(best_id)
            best_id_top3_total.append(best_id_top3)
            best_id_top5_total.append(best_id_top5)
            best_efield_total.append(best_efield)
            best_efield_top3_total.append(best_efield_top3)
            best_efield_top5_total.append(best_efield_top5)
            best_pos_total.append(best_pos)
            best_angle_total.append(best_angle)
        np.save(os.path.join(output_root_path,subject.split('_')[-1]+'_'+k+'_best_id.npy'),np.array(best_id_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_id_top3.npy'),np.array(best_id_top3_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_id_top5.npy'),np.array(best_id_top5_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_efield.npy'), np.array(best_efield_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_efield_top3.npy'), np.array(best_efield_top3_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_efield_top5.npy'), np.array(best_efield_top5_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_pos.npy'), np.array(best_pos_total))
        np.save(os.path.join(output_root_path, subject.split('_')[-1] + '_' + k + '_best_angle.npy'), np.array(best_angle_total))
# print('best_id_match_percent: ',best_id_match/total_num)
# print('best_id_match_correction_percent: ',best_id_match_correction/total_num)