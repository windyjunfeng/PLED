# coding=utf-8

import os
import numpy as np
import pandas as pd

subjects=['m2m_A2','m2m_A3','m2m_A8','m2m_A9','m2m_A14','m2m_A17','m2m_A18','m2m_A20','m2m_A24']
# methods_folders={'Origin':'35','Baseline':'31_based_step1','PLED (Ours)':'39_based_step1'}
# methods_folders={'Origin':'35_add_lesion','Baseline':'31_based_step1_add_lesion','PLED (Ours)':'39_based_step1_add_lesion'}
# methods_folders={'Origin':'35_optimization_angleres_1','Baseline':'31_based_gtlabel_optimization_angleres_1','PLED':'39_based_gtlabel_optimization_angleres_1'}
# methods_folders={'PLED_depth_32':'39_based_gtlabel_optimization','PLED_depth_64':'43_3_based_gtlabel_optimization_depth_64'}
# subjects=['m2m_390645','m2m_432332','m2m_566454','m2m_644044','m2m_753150']
# subjects=['m2m_B2','m2m_B4','m2m_B13','m2m_B16','m2m_B23']
# subjects=['m2m_A2','m2m_A9']
# subjects=['m2m_TMS-015','m2m_TMS-040','m2m_TMS-076','m2m_TMS-103']
# subjects=['m2m_753150']
# methods_folders={'Origin':'35_cog','Baseline':'31_based_step1_cog','PLED':'39_based_step1_cog'}
# methods_folders={'gt_depth_64':'gt_optimization_depth_64'}
methods_folders={'gt_vector':'gt_optimization_vector'}
classnames={'gm':'GM','foreground':'Total'}
aspects={'pos':'Distance of optimal coil positions (mm)','angle':'MAE of optimal coil orientation (°)','efield':'MRE of optimal E-field magnitude (%)'}
# aspects={'angle':'MAE of optimal coil orientation (°)','efield':'MRE of optimal E-field magnitude (%)'}
input_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health'
output_root_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/results_analysis'
# sub_foldername=r'box_plot_viridis_nolegend\total'
# sub_foldername=r'box_plot_viridis_nolegend_39_based_gt_step1\total'
# gt_foldername=r'gt'
gt_foldername=r'gt_optimization'
sub_foldername=r'efield_scalar_vector_gt_analysis'
output_path=os.path.join(output_root_path,sub_foldername)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def append_value(total,item):
    if total is None:
        total=item.copy()
    else:
        total=np.concatenate((total,item))
    return total


# 存储统计结果
results = {}

for classname in classnames.keys():
    results[classname] = {}
    
    for aspect in aspects.keys():
        results[classname][aspect] = {}
        
        for method in methods_folders.keys():
            values = None

            for subject in subjects:
                # if method=='Origin' and subject not in ['m2m_A2','m2m_A9']:
                #     continue

                gt_path=os.path.join(input_root_path,gt_foldername,'optimal_eval',subject.split('_')[-1]+'_'+classname+'_best_'+aspect+'.npy')
                gt_data=np.load(gt_path)
                predict_path=os.path.join(input_root_path,methods_folders[method],'optimal_eval',subject.split('_')[-1]+'_'+classname+'_best_'+aspect+'.npy')
                predict_data=np.load(predict_path)
                if aspect=='efiled':
                    eval=np.abs(predict_data-gt_data)/gt_data  # MRE
                elif aspect=='pos':
                    eval=np.linalg.norm((predict_data-gt_data),axis=1)  # l2距离
                else:
                    eval=np.abs(predict_data-gt_data) # MAE
                mask=np.isfinite(eval)
                values_filtered=eval[mask]  # 过滤掉为nan或inf的元素
                values=append_value(values,values_filtered)
            
            # 计算均值和标准差
            if values is not None and len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                results[classname][aspect][method] = {
                    'mean': mean_val,
                    'std': std_val,
                    'count': len(values)
                }
            else:
                results[classname][aspect][method] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'count': 0
                }

# 打印结果
print("=" * 80)
print("Aspects Statistics (Mean ± Std)")
print("=" * 80)

for classname in classnames.keys():
    print(f"\n{classnames[classname]} ({classname}):")
    print("-" * 80)
    
    for aspect in aspects.keys():
        print(f"\n  {aspects[aspect]} ({aspect}):")
        for method in methods_folders.keys():
            if method in results[classname][aspect]:
                stats = results[classname][aspect][method]
                print(f"    {method:20s}: Mean = {stats['mean']:10.6f}, Std = {stats['std']:10.6f}, N = {stats['count']}")

# 保存结果到CSV文件
for classname in classnames.keys():
    output_class_path = os.path.join(output_path, classname)
    if not os.path.exists(output_class_path):
        os.makedirs(output_class_path)
    
    # 为每个aspect创建CSV文件
    for aspect in aspects.keys():
        data_rows = []
        for method in methods_folders.keys():
            if method in results[classname][aspect]:
                stats = results[classname][aspect][method]
                data_rows.append({
                    'Method': method,
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'Count': stats['count']
                })
        
        df = pd.DataFrame(data_rows)
        csv_path = os.path.join(output_class_path, f'{aspect}_stats.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

print("\n" + "=" * 80)
print("Calculation completed!")
print("=" * 80)
