# coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size':15})


def filter_files(directory, first_char, second_char):
    """
    根据目录路径下的文件名，以'_'为分割符，筛选第一个和第二个字符为指定字符的文件
    
    Args:
        directory: 目录路径
        first_char: 第一个字符（分割后的第一个字段的第一个字符）
        second_char: 第二个字符（分割后的第二个字段的第一个字符）
    
    Returns:
        符合条件的文件路径列表
    """
    filtered_files = []
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return filtered_files
    
    for filename in os.listdir(directory):
        if not filename.endswith('.npy'):
            continue
        
        # 以'_'分割文件名（不包括扩展名）
        prefilename, _ = os.path.splitext(filename)
        parts = prefilename.split('_')
        
        # 检查第一个和第二个字段的第一个字符
        if len(parts) >= 2:
            if len(parts[0]) > 0 and len(parts[1]) > 0:
                if parts[0] in first_char and parts[1] in second_char:
                    filepath = os.path.join(directory, filename)
                    filtered_files.append(filepath)
    
    return filtered_files


def extract_mask_category(filepath):
    """
    从文件路径中提取mask_category（第二个字段）
    
    Args:
        filepath: 文件路径
    
    Returns:
        mask_category字符串，如果无法提取则返回None
    """
    filename = os.path.basename(filepath)
    prefilename, _ = os.path.splitext(filename)
    parts = prefilename.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None


def plot_npy_data(npy_files, output_dir=None):
    """
    读取npy文件并分别绘制散点图和条形图（分开保存）
    
    Args:
        npy_files: npy文件路径列表
        output_dir: 输出目录，如果为None则不保存图片
    """
    if len(npy_files) == 0:
        print("没有找到符合条件的文件")
        return
    
    # 横轴：-90到89
    x_axis = np.arange(-90, 90)
    
    # 为每个文件创建图形
    for filepath in npy_files:
        # 读取数据
        data = np.load(filepath)
        
        if data.ndim != 1:
            print(f"警告: {filepath} 不是一维数组，跳过")
            continue
        
        # 检查数据长度是否为180的倍数
        if len(data) % 180 != 0:
            print(f"警告: {filepath} 数据长度 {len(data)} 不是180的倍数，跳过")
            continue
        
        # 将数据重塑为 (n_groups, 180) 的形状
        n_groups = len(data) // 180
        data_reshaped = data.reshape(n_groups, 180)
        
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 绘制散点图：每组180个元素中值最大的元素
        # 对于每一组，找到该组中值最大的元素（即该组的最大值），绘制该最大值对应的角度和值
        max_angles = []
        max_values = []
        for i in range(n_groups):
            group_data = data_reshaped[i, :]
            max_idx = np.argmax(group_data)  # 找到该组最大值的索引
            max_angle = x_axis[max_idx]  # 对应的角度
            max_value = group_data[max_idx]  # 最大值
            max_angles.append(max_angle)
            max_values.append(max_value)
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
        ax_scatter.scatter(max_angles, max_values, alpha=0.7, s=20)
        
        ax_scatter.set_xlabel('Direction (°)', fontsize=15)
        ax_scatter.set_ylabel('E-field magnitude (V/m)', fontsize=15)
        
        if output_dir is not None:
            scatter_output_path = os.path.join(output_dir, base_name + '_scatter.png')
            plt.savefig(scatter_output_path, dpi=1200, bbox_inches='tight')
            print(f"散点图已保存: {scatter_output_path}")
        plt.close(fig_scatter)
        
        # 绘制条形图：每个角度对应的最大元素的个数
        # 对于每个角度，统计有多少组在该角度达到最大值（即该角度是该组的最大值位置）
        max_count_per_angle = np.zeros(180)
        for i in range(n_groups):
            group_data = data_reshaped[i, :]
            max_idx = np.argmax(group_data)  # 找到该组最大值的索引
            max_count_per_angle[max_idx] += 1
        
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        ax_bar.bar(x_axis, max_count_per_angle, width=1.0, alpha=0.7)
        
        ax_bar.set_xlabel('Direction (°)', fontsize=15)
        ax_bar.set_ylabel('Count of groups with maximum at this angle', fontsize=15)
        
        if output_dir is not None:
            bar_output_path = os.path.join(output_dir, base_name + '_bar.png')
            plt.savefig(bar_output_path, dpi=1200, bbox_inches='tight')
            print(f"条形图已保存: {bar_output_path}")
        plt.close(fig_bar)


def plot_angle_resolution_analysis(npy_files, output_dir=None):
    """
    分析不同角度分辨率下的元素值变化
    
    Args:
        npy_files: npy文件路径列表
        output_dir: 输出目录，如果为None则不保存图片
    """
    if len(npy_files) == 0:
        print("没有找到符合条件的文件")
        return
    
    # 角度分辨率从1°到20°
    angle_resolutions = np.arange(1, 21)
    
    # 为每个文件创建图形
    for filepath in npy_files:
        # 读取数据
        data = np.load(filepath)
        
        if data.ndim != 1:
            print(f"警告: {filepath} 不是一维数组，跳过")
            continue
        
        # 检查数据长度是否为180的倍数
        if len(data) % 180 != 0:
            print(f"警告: {filepath} 数据长度 {len(data)} 不是180的倍数，跳过")
            continue
        
        # 将数据重塑为 (n_groups, 180) 的形状
        n_groups = len(data) // 180
        data_reshaped = data.reshape(n_groups, 180)
        
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 存储每个分辨率下的统计值（波动范围的统计值）
        mean_ranges = []
        std_ranges = []
        min_ranges = []
        max_ranges = []
        
        for resolution in angle_resolutions:
            # 计算该分辨率下的分组数
            n_bins = 180 // resolution
            
            # 存储每个小组的波动范围
            bin_ranges = []
            
            for group_idx in range(n_groups):
                group_data = data_reshaped[group_idx, :]
                
                # 将180个角度按照分辨率分成若干小组
                for bin_idx in range(n_bins):
                    start_idx = bin_idx * resolution
                    end_idx = min(start_idx + resolution, 180)
                    bin_data = group_data[start_idx:end_idx]
                    # 计算该小组内元素值的最大波动范围（最大值-最小值）
                    bin_range = np.max(bin_data) - np.min(bin_data)
                    bin_ranges.append(bin_range)
            
            # 计算所有小组波动范围的统计值
            bin_ranges = np.array(bin_ranges)
            mean_ranges.append(np.mean(bin_ranges))
            std_ranges.append(np.std(bin_ranges))
            min_ranges.append(np.min(bin_ranges))
            max_ranges.append(np.max(bin_ranges))
        
        mean_ranges = np.array(mean_ranges)
        std_ranges = np.array(std_ranges)
        min_ranges = np.array(min_ranges)
        max_ranges = np.array(max_ranges)
        
        # 绘制折线图（体现均值加波动范围）
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制均值线
        ax.plot(angle_resolutions, mean_ranges, 'b-', linewidth=2, marker='o', markersize=6, label='Mean Range')
        
        # 绘制波动范围（均值±标准差）
        ax.fill_between(angle_resolutions, 
                        mean_ranges - std_ranges, 
                        mean_ranges + std_ranges, 
                        alpha=0.3, color='blue', label='Mean ± Std')
        
        # 绘制最小值和最大值范围
        ax.fill_between(angle_resolutions, 
                        min_ranges, 
                        max_ranges, 
                        alpha=0.2, color='gray', label='Min-Max Range')
        
        ax.set_xlabel('Angle Resolution (°)', fontsize=15)
        ax.set_ylabel('Fluctuation Range (V/m)', fontsize=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_dir is not None:
            resolution_output_path = os.path.join(output_dir, base_name + '_angle_resolution.png')
            plt.savefig(resolution_output_path, dpi=1200, bbox_inches='tight')
            print(f"角度分辨率分析图已保存: {resolution_output_path}")
        plt.close(fig)


def plot_combined_by_category(npy_files, output_dir=None):
    """
    将相同mask_category的文件合并绘制散点图、条形图和角度分辨率分析图
    
    Args:
        npy_files: npy文件路径列表
        output_dir: 输出目录，如果为None则不保存图片
    """
    if len(npy_files) == 0:
        print("没有找到符合条件的文件")
        return
    
    # 按mask_category分组
    files_by_category = {}
    for filepath in npy_files:
        category = extract_mask_category(filepath)
        if category is None:
            print(f"警告: 无法从 {filepath} 提取mask_category，跳过")
            continue
        if category not in files_by_category:
            files_by_category[category] = []
        files_by_category[category].append(filepath)
    
    # 横轴：-90到89
    x_axis = np.arange(-90, 90)
    # 角度分辨率从1°到20°
    angle_resolutions = np.arange(1, 21)
    
    # 为每个category绘制合并图
    for category, file_list in files_by_category.items():
        all_data_reshaped = []
        
        # 读取所有文件的数据
        for filepath in file_list:
            data = np.load(filepath)
            
            if data.ndim != 1:
                print(f"警告: {filepath} 不是一维数组，跳过")
                continue
            
            if len(data) % 180 != 0:
                print(f"警告: {filepath} 数据长度 {len(data)} 不是180的倍数，跳过")
                continue
            
            n_groups = len(data) // 180
            data_reshaped = data.reshape(n_groups, 180)
            all_data_reshaped.append(data_reshaped)
        
        if len(all_data_reshaped) == 0:
            continue
        
        # 合并所有数据
        combined_data = np.concatenate(all_data_reshaped, axis=0)  # 沿第一个维度合并
        
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 绘制合并的散点图：每组180个元素中值最大的元素
        # 对于每一组，找到该组中值最大的元素（即该组的最大值），绘制该最大值对应的角度和值
        n_groups_combined = combined_data.shape[0]
        max_angles = []
        max_values = []
        for i in range(n_groups_combined):
            group_data = combined_data[i, :]
            max_idx = np.argmax(group_data)
            max_angle = x_axis[max_idx]
            max_value = group_data[max_idx]
            max_angles.append(max_angle)
            max_values.append(max_value)
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
        ax_scatter.scatter(max_angles, max_values, alpha=0.7, s=20)
        
        ax_scatter.set_xlabel('Direction (°)', fontsize=15)
        ax_scatter.set_ylabel('E-field magnitude (V/m)', fontsize=15)
        
        scatter_output_path = os.path.join(output_dir, f'{category}_combined_scatter.png')
        plt.savefig(scatter_output_path, dpi=1200, bbox_inches='tight')
        print(f"合并散点图已保存: {scatter_output_path}")
        plt.close(fig_scatter)
        
        # 绘制合并的条形图：每个角度对应的最大元素的个数
        max_count_per_angle = np.zeros(180)
        for i in range(n_groups_combined):
            group_data = combined_data[i, :]
            max_idx = np.argmax(group_data)
            max_count_per_angle[max_idx] += 1
        
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        ax_bar.bar(x_axis, max_count_per_angle, width=1.0, alpha=0.7)
        
        ax_bar.set_xlabel('Direction (°)', fontsize=15)
        ax_bar.set_ylabel('Count of groups with maximum at this angle', fontsize=15)
        
        bar_output_path = os.path.join(output_dir, f'{category}_combined_bar.png')
        plt.savefig(bar_output_path, dpi=1200, bbox_inches='tight')
        print(f"合并条形图已保存: {bar_output_path}")
        plt.close(fig_bar)
        
        # 绘制合并的角度分辨率分析图
        mean_ranges = []
        std_ranges = []
        min_ranges = []
        max_ranges = []
        
        for resolution in angle_resolutions:
            n_bins = 180 // resolution
            bin_ranges = []
            
            for group_idx in range(n_groups_combined):
                group_data = combined_data[group_idx, :]
                for bin_idx in range(n_bins):
                    start_idx = bin_idx * resolution
                    end_idx = min(start_idx + resolution, 180)
                    bin_data = group_data[start_idx:end_idx]
                    # 计算该小组内元素值的最大波动范围（最大值-最小值）
                    bin_range = np.max(bin_data) - np.min(bin_data)
                    bin_ranges.append(bin_range)
            
            bin_ranges = np.array(bin_ranges)
            mean_ranges.append(np.mean(bin_ranges))
            std_ranges.append(np.std(bin_ranges))
            min_ranges.append(np.min(bin_ranges))
            max_ranges.append(np.max(bin_ranges))
        
        mean_ranges = np.array(mean_ranges)
        std_ranges = np.array(std_ranges)
        min_ranges = np.array(min_ranges)
        max_ranges = np.array(max_ranges)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(angle_resolutions, mean_ranges, 'b-', linewidth=2, marker='o', markersize=6, label='Mean Range')
        ax.fill_between(angle_resolutions, 
                        mean_ranges - std_ranges, 
                        mean_ranges + std_ranges, 
                        alpha=0.3, color='blue', label='Mean ± Std')
        ax.fill_between(angle_resolutions, 
                        min_ranges, 
                        max_ranges, 
                        alpha=0.2, color='gray', label='Min-Max Range')
        
        ax.set_xlabel('Angle Resolution (°)', fontsize=15)
        ax.set_ylabel('Fluctuation Range (V/m)', fontsize=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        resolution_output_path = os.path.join(output_dir, f'{category}_combined_angle_resolution.png')
        plt.savefig(resolution_output_path, dpi=1200, bbox_inches='tight')
        print(f"合并角度分辨率分析图已保存: {resolution_output_path}")
        plt.close(fig)


def main():
    """
    主函数：示例用法
    """
    # 示例参数
    directory = r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/gt_optimization_angleres_1/mean_efield'  # 修改为你的目录路径
    subjects=['A2','A3','A8','A9','A14','A17','A18','A20','A24']
    mask_categories = ['gm','foreground']  # 修改为第二个字符
    output_dir = r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/gt_optimization_angleres_1/mean_efield/mean_efield_analysis'  # 输出目录，如果为None则不保存
    
    # 筛选文件
    filtered_files = filter_files(directory, subjects, mask_categories)
    print(f"找到 {len(filtered_files)} 个符合条件的文件:")
    for f in filtered_files:
        print(f"  - {f}")
    
    # 绘制单个文件的散点图和条形图（分开保存）
    if len(filtered_files) > 0:
        plot_npy_data(filtered_files, output_dir)
    
    # 绘制单个文件的角度分辨率分析图
    if len(filtered_files) > 0:
        plot_angle_resolution_analysis(filtered_files, output_dir)
    
    # 绘制相同mask_category的合并图
    if len(filtered_files) > 0:
        plot_combined_by_category(filtered_files, output_dir)


if __name__ == '__main__':
    main()
