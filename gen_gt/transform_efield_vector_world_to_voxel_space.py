import os
import json
import numpy as np
from tqdm import tqdm
import argparse


def transform_vector_world_to_voxel(efield_world, affine_matrix):
    """
    将世界坐标系下的电场矢量转换到体素坐标系
    
    参数:
    - efield_world: numpy数组，形状为(..., 3)，世界坐标系下的电场矢量
    - affine_matrix: numpy数组，形状为(4, 4)，从体素坐标到世界坐标的affine变换矩阵
    
    返回:
    - efield_voxel: numpy数组，形状与efield_world相同，体素坐标系下的电场矢量
    """
    # 提取旋转矩阵部分（3x3）
    R = affine_matrix[:3, :3]
    
    # 对于向量，只需要旋转，不需要平移
    # 从世界到体素的变换是R的逆矩阵
    R_inv = np.linalg.inv(R)
    
    # 获取原始形状
    original_shape = efield_world.shape
    # 将电场矢量重塑为(N, 3)以便进行矩阵乘法
    efield_reshaped = efield_world.reshape(-1, 3)
    
    # 应用旋转矩阵：E_voxel = R_inv @ E_world
    # 由于efield_reshaped是行向量形式(N, 3)，需要右乘R_inv的转置
    # 这等价于对列向量形式左乘R_inv
    efield_voxel_reshaped = efield_reshaped @ R_inv.T
    
    # 恢复原始形状
    efield_voxel = efield_voxel_reshaped.reshape(original_shape)
    
    return efield_voxel


def process_directory(input_dir, json_path, output_dir, verbose=True):
    """
    处理目录中的所有npy文件，将电场矢量从世界坐标系转换到体素坐标系
    
    参数:
    - input_dir: 输入npy文件目录路径
    - json_path: 存储affine矩阵的json文件路径
    - output_dir: 输出目录路径
    - verbose: 是否打印详细信息
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取affine矩阵
    with open(json_path, 'r') as f:
        affine_matrices = json.load(f)
    
    # 获取所有npy文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        if verbose:
            print(f"警告: 在目录 {input_dir} 中未找到npy文件")
        return 0
    
    if verbose:
        print(f"找到 {len(npy_files)} 个npy文件")
    
    success_count = 0
    # 处理每个文件
    for npy_file in tqdm(npy_files, desc="转换电场矢量", disable=not verbose):
        # 获取文件名（不含扩展名）作为键
        file_key = os.path.splitext(npy_file)[0]
        
        # 检查是否有对应的affine矩阵
        if file_key not in affine_matrices:
            if verbose:
                print(f"警告: 文件 {npy_file} 没有对应的affine矩阵，跳过")
            continue
        
        # 读取npy文件
        input_path = os.path.join(input_dir, npy_file)
        efield_world = np.load(input_path)
        
        # 检查数据形状
        if efield_world.shape[-1] != 3:
            if verbose:
                print(f"警告: 文件 {npy_file} 的最后一个维度不是3，形状为 {efield_world.shape}，跳过")
            continue
        
        # 获取对应的affine矩阵
        affine_matrix = np.array(affine_matrices[file_key])
        
        # 检查affine矩阵形状
        if affine_matrix.shape != (4, 4):
            if verbose:
                print(f"警告: 文件 {npy_file} 的affine矩阵形状不正确，为 {affine_matrix.shape}，跳过")
            continue
        
        # 转换电场矢量
        efield_voxel = transform_vector_world_to_voxel(efield_world, affine_matrix)
        
        # 保存转换后的文件
        output_path = os.path.join(output_dir, npy_file)
        np.save(output_path, efield_voxel.astype(np.float16))
        success_count += 1
    
    if verbose:
        print(f"转换完成！成功处理 {success_count}/{len(npy_files)} 个文件，结果保存在 {output_dir}")
    
    return success_count


def process_root_directory(root_dir, input_dir_rel, json_path_rel, output_dir_rel):
    """
    批量处理根目录下所有子文件夹
    
    参数:
    - root_dir: 根目录路径
    - input_dir_rel: 相对于子文件夹的输入目录路径
    - json_path_rel: 相对于子文件夹的json文件路径
    - output_dir_rel: 相对于子文件夹的输出目录路径
    """
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, d))]
    
    if len(subdirs) == 0:
        print(f"警告: 在根目录 {root_dir} 中未找到子文件夹")
        return
    
    print(f"找到 {len(subdirs)} 个子文件夹")
    
    total_files = 0
    total_failed_dirs = 0
    
    for subdir in tqdm(subdirs, desc="处理子文件夹"):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 构建完整路径
        input_dir = os.path.join(subdir_path, input_dir_rel)
        json_path = os.path.join(subdir_path, json_path_rel)
        output_dir = os.path.join(subdir_path, output_dir_rel)
        
        # 检查路径是否存在
        if not os.path.exists(input_dir):
            print(f"警告: 子文件夹 {subdir} 的输入目录不存在: {input_dir}，跳过")
            total_failed_dirs += 1
            continue
        
        if not os.path.exists(json_path):
            print(f"警告: 子文件夹 {subdir} 的JSON文件不存在: {json_path}，跳过")
            total_failed_dirs += 1
            continue
        
        # 处理该子文件夹
        print(f"\n处理子文件夹: {subdir}")
        try:
            count = process_directory(input_dir, json_path, output_dir, verbose=True)
            total_files += count
        except Exception as e:
            print(f"错误: 处理子文件夹 {subdir} 时出错: {e}")
            total_failed_dirs += 1
    
    print(f"\n批量处理完成！")
    print(f"成功处理 {len(subdirs) - total_failed_dirs}/{len(subdirs)} 个子文件夹")
    print(f"总共成功转换 {total_files} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description='将电场矢量从世界坐标系转换到体素坐标系'
    )
    
    # 单目录模式参数
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='输入npy文件目录路径（单目录模式）'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default=None,
        help='存储affine矩阵的json文件路径（单目录模式）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录路径（单目录模式）'
    )
    
    # 批量处理模式参数
    parser.add_argument(
        '--root_dir',
        type=str,
        default=None,
        help='根目录路径（批量处理模式）'
    )
    parser.add_argument(
        '--input_dir_rel',
        type=str,
        default=None,
        help='相对于子文件夹的输入目录路径（批量处理模式）'
    )
    parser.add_argument(
        '--json_path_rel',
        type=str,
        default=None,
        help='相对于子文件夹的json文件路径（批量处理模式）'
    )
    parser.add_argument(
        '--output_dir_rel',
        type=str,
        default=None,
        help='相对于子文件夹的输出目录路径（批量处理模式）'
    )
    
    args = parser.parse_args()
    
    # 判断使用哪种模式
    if args.root_dir is not None:
        # 批量处理模式
        if args.input_dir_rel is None or args.json_path_rel is None or args.output_dir_rel is None:
            raise ValueError("批量处理模式需要指定 --input_dir_rel, --json_path_rel 和 --output_dir_rel")
        
        if not os.path.exists(args.root_dir):
            raise ValueError(f"根目录不存在: {args.root_dir}")
        
        process_root_directory(args.root_dir, args.input_dir_rel, args.json_path_rel, args.output_dir_rel)
    
    else:
        # 单目录模式
        if args.input_dir is None or args.json_path is None or args.output_dir is None:
            raise ValueError("单目录模式需要指定 --input_dir, --json_path 和 --output_dir")
        
        # 检查输入目录是否存在
        if not os.path.exists(args.input_dir):
            raise ValueError(f"输入目录不存在: {args.input_dir}")
        
        # 检查json文件是否存在
        if not os.path.exists(args.json_path):
            raise ValueError(f"JSON文件不存在: {args.json_path}")
        
        # 处理目录
        process_directory(args.input_dir, args.json_path, args.output_dir)


if __name__ == '__main__':
    main()

