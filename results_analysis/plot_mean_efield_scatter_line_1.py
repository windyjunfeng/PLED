# coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm

rcParams.update({'font.size':15})

# 每条线使用不同颜色：使用 tab20 色图，最多 20 条不重复，超过则用 viridis 采样
def get_line_colors(n_lines):
    if n_lines <= 20:
        return [cm.get_cmap('tab20')(i) for i in range(n_lines)]
    return [cm.get_cmap('viridis')(i / max(n_lines - 1, 1)) for i in range(n_lines)]


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


def plot_npy_data(npy_files, groups, output_dir=None):
    """
    读取npy文件并分别绘制折线图（无标记点），按groups图例，每1/4的groups拆成一张图；以及均值折线图（无标记点）。
    
    Args:
        npy_files: npy文件路径列表
        groups: 图例名称列表，与各条折线顺序对应
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
        # 图例用 groups 中对应顺序的名称，不足时用 Group i+1
        legend_labels = [groups[i] if i < len(groups) else f'Group {i+1}' for i in range(n_groups)]
        
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 折线图拆成 4 张图，每张约 1/4 的 groups，每条线不同颜色
        n_figs = 4
        chunk_size = (n_groups + n_figs - 1) // n_figs
        for fig_idx in range(n_figs):
            start_i = fig_idx * chunk_size
            end_i = min(start_i + chunk_size, n_groups)
            if start_i >= n_groups:
                break
            n_lines_part = end_i - start_i
            colors_part = get_line_colors(n_lines_part)
            fig_part, ax_part = plt.subplots(figsize=(12, 6))
            for k, i in enumerate(range(start_i, end_i)):
                y_values = data_reshaped[i, :]
                ax_part.plot(x_axis, y_values, color=colors_part[k], alpha=0.8, label=legend_labels[i])
            ax_part.set_xlabel('Direction (°)', fontsize=15)
            ax_part.set_ylabel('E-field magnitude (V/m)', fontsize=15)
            ax_part.legend()
            if output_dir is not None:
                part_output_path = os.path.join(output_dir, f'{base_name}_line_part{fig_idx + 1}.png')
                plt.savefig(part_output_path, dpi=1200, bbox_inches='tight')
                print(f"折线图(部分{fig_idx + 1})已保存: {part_output_path}")
            plt.close(fig_part)
        
        # 绘制均值折线图（无标记点）
        fig_line, ax_line = plt.subplots(figsize=(12, 6))
        mean_values = np.mean(data_reshaped, axis=0)
        ax_line.plot(x_axis, mean_values, 'r-', linewidth=2)
        
        ax_line.set_xlabel('Direction (°)', fontsize=15)
        ax_line.set_ylabel('E-field magnitude (V/m)', fontsize=15)
        
        if output_dir is not None:
            line_output_path = os.path.join(output_dir, base_name + '_line.png')
            plt.savefig(line_output_path, dpi=1200, bbox_inches='tight')
            print(f"均值折线图已保存: {line_output_path}")
        plt.close(fig_line)


def plot_combined_by_category(npy_files, groups, output_dir=None):
    """
    将相同mask_category的文件合并绘制折线图（无标记点），按groups图例，每1/4的groups拆成一张图；以及均值折线图（无标记点）。
    
    Args:
        npy_files: npy文件路径列表
        groups: 图例名称列表，与各条折线顺序对应
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
        
        # 合并：每个 group 对所有个体的该条折线求平均，得到 (n_groups, 180)，即每张图仍是 12 条线，每条线是所有个体该 group 的平均
        stacked = np.stack(all_data_reshaped, axis=0)  # (n_subjects, n_groups, 180)
        combined_data = np.mean(stacked, axis=0)  # (n_groups, 180)
        n_groups = combined_data.shape[0]
        legend_labels = [groups[i] if i < len(groups) else f'Group {i+1}' for i in range(n_groups)]
        
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # 折线图拆成 4 张图，每张 12 条线，图例为对应 groups，每条线不同颜色
        n_figs = 4
        chunk_size = (n_groups + n_figs - 1) // n_figs
        for fig_idx in range(n_figs):
            start_i = fig_idx * chunk_size
            end_i = min(start_i + chunk_size, n_groups)
            if start_i >= n_groups:
                break
            n_lines_part = end_i - start_i
            colors_part = get_line_colors(n_lines_part)
            fig_part, ax_part = plt.subplots(figsize=(12, 6))
            for k, i in enumerate(range(start_i, end_i)):
                y_values = combined_data[i, :]
                ax_part.plot(x_axis, y_values, color=colors_part[k], alpha=0.8, label=legend_labels[i])
            ax_part.set_xlabel('Direction (°)', fontsize=15)
            ax_part.set_ylabel('E-field magnitude (V/m)', fontsize=15)
            ax_part.legend()
            if output_dir is not None:
                part_output_path = os.path.join(output_dir, f'{category}_combined_line_part{fig_idx + 1}.png')
                plt.savefig(part_output_path, dpi=1200, bbox_inches='tight')
                print(f"合并折线图(部分{fig_idx + 1})已保存: {part_output_path}")
            plt.close(fig_part)
        
        # 绘制合并的均值折线图（无标记点）
        fig_line, ax_line = plt.subplots(figsize=(12, 6))
        mean_values = np.mean(combined_data, axis=0)
        ax_line.plot(x_axis, mean_values, 'r-', linewidth=2)
        
        ax_line.set_xlabel('Direction (°)', fontsize=15)
        ax_line.set_ylabel('E-field magnitude (V/m)', fontsize=15)
        
        if output_dir is not None:
            line_output_path = os.path.join(output_dir, f'{category}_combined_line.png')
            plt.savefig(line_output_path, dpi=1200, bbox_inches='tight')
            print(f"合并均值折线图已保存: {line_output_path}")
        plt.close(fig_line)


def main():
    """
    主函数：示例用法
    """
    # 示例参数
    directory = r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/gt_optimization_angleres_1/mean_efield'  # 修改为你的目录路径
    subjects=['A2','A3','A8','A9','A14','A17','A18','A20','A24']
    groups=['AF3', 'AFz', 'AF4', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
    mask_categories = ['gm','foreground']  # 修改为第二个字符
    output_dir = r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/gt_optimization_angleres_1/mean_efield/mean_efield_analysis_1'  # 输出目录，如果为None则不保存
    
    # 筛选文件
    filtered_files = filter_files(directory, subjects, mask_categories)
    print(f"找到 {len(filtered_files)} 个符合条件的文件:")
    for f in filtered_files:
        print(f"  - {f}")
    
    # 绘制单个文件的折线图（分4张）和均值折线图
    if len(filtered_files) > 0:
        plot_npy_data(filtered_files, groups, output_dir)
    
    # 绘制相同mask_category的合并折线图（分4张）和均值折线图
    if len(filtered_files) > 0:
        plot_combined_by_category(filtered_files, groups, output_dir)


if __name__ == '__main__':
    main()
