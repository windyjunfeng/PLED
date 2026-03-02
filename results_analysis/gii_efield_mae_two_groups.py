# coding=utf-8
"""
读取两组 label_efield_reconstruction_gii 输出路径下的 gii 文件：
  root_path_1/subject/output_relpath_1/sampling_name/ 与
  root_path_2/subject/output_relpath_2/sampling_name/
对相同 subject、sampling_name 下的 magnitude 和 normal_component 做相减取绝对值（MAE），
保存到第一组路径下：root_path_1/subject/output_relpath_1/sampling_name/*_mae.shape.gii
"""
import os
import numpy as np
import nibabel as nib
import nibabel.gifti as gif


def compute_and_save_gii_mae(
    root_path_1,
    root_path_2,
    output_relpath_1,
    output_relpath_2,
    subjects=None,
):
    """
    从两组路径下读取：
    第一组：root_path_1 / subject / output_relpath_1 / sampling_name / *.shape.gii
    第二组：root_path_2 / subject / output_relpath_2 / sampling_name / *.shape.gii
    对相同 subject、相同 sampling_name 下的 magnitude 和 normal_component 做 |data1 - data2|，
    保存到第一组路径下：root_path_1 / subject / output_relpath_1 / sampling_name / *_mae.shape.gii

    Args:
        root_path_1: 第一组根路径（其下为 subject 文件夹），且 MAE 结果也写在此路径下
        root_path_2: 第二组根路径
        output_relpath_1: 第一组相对路径，如 'visualization/local_sampling_scalp_labels_new_1010_center_efield_vector'
        output_relpath_2: 第二组相对路径（可与 output_relpath_1 相同或不同）
        subjects: 可选，指定 subject 列表；为 None 时取两组路径下均有对应 output_relpath 的 subject 交集
    """
    rel_1 = output_relpath_1.replace("\\", "/").rstrip("/")
    rel_2 = output_relpath_2.replace("\\", "/").rstrip("/")

    # 确定 subject 列表
    if subjects is None:
        subdirs_1 = set()
        subdirs_2 = set()
        for name in os.listdir(root_path_1):
            p = os.path.join(root_path_1, name, rel_1)
            if os.path.isdir(p):
                subdirs_1.add(name)
        for name in os.listdir(root_path_2):
            p = os.path.join(root_path_2, name, rel_2)
            if os.path.isdir(p):
                subdirs_2.add(name)
        subjects = sorted(subdirs_1 & subdirs_2)
        if not subjects:
            print("未找到两组共有的 subject（在各自 output_relpath 下），退出")
            return
    else:
        subjects = list(subjects)

    suffix_mag = "_efield_vector_magnitude.shape.gii"
    suffix_norm = "_efield_vector_normal_component.shape.gii"
    out_suffix_mag = "_efield_vector_magnitude_mae.shape.gii"
    out_suffix_norm = "_efield_vector_normal_component_mae.shape.gii"

    for subject in subjects:
        path_rel_1 = os.path.join(root_path_1, subject, rel_1)
        path_rel_2 = os.path.join(root_path_2, subject, rel_2)
        if not os.path.isdir(path_rel_1) or not os.path.isdir(path_rel_2):
            continue

        # 列出两边都存在的 sampling 子文件夹（取交集）
        try:
            names_1 = {d for d in os.listdir(path_rel_1) if os.path.isdir(os.path.join(path_rel_1, d))}
            names_2 = {d for d in os.listdir(path_rel_2) if os.path.isdir(os.path.join(path_rel_2, d))}
            sampling_names = sorted(names_1 & names_2)
        except OSError:
            sampling_names = []

        for sampling_name in sampling_names:
            f1_mag = os.path.join(path_rel_1, sampling_name, sampling_name + suffix_mag)
            f1_norm = os.path.join(path_rel_1, sampling_name, sampling_name + suffix_norm)
            f2_mag = os.path.join(path_rel_2, sampling_name, sampling_name + suffix_mag)
            f2_norm = os.path.join(path_rel_2, sampling_name, sampling_name + suffix_norm)
            if not all(os.path.isfile(p) for p in (f1_mag, f1_norm, f2_mag, f2_norm)):
                continue

            # 读取并计算 MAE
            def load_darray(path):
                img = nib.load(path)
                return np.asarray(img.darrays[0].data, dtype=np.float64)

            d1_mag = load_darray(f1_mag)
            d2_mag = load_darray(f2_mag)
            d1_norm = load_darray(f1_norm)
            d2_norm = load_darray(f2_norm)

            if d1_mag.shape != d2_mag.shape or d1_norm.shape != d2_norm.shape:
                print(f"跳过 {subject}/{sampling_name}: 两组数据形状不一致")
                continue

            mae_mag = np.abs(d1_mag - d2_mag)
            mae_norm = np.abs(d1_norm - d2_norm)

            # 保存到第一组路径下：root_path_1 / subject / output_relpath_1 / sampling_name /
            out_dir = os.path.join(path_rel_1, sampling_name)
            os.makedirs(out_dir, exist_ok=True)
            out_mag = os.path.join(out_dir, sampling_name + out_suffix_mag)
            out_norm_path = os.path.join(out_dir, sampling_name + out_suffix_norm)

            gii_mag = gif.GiftiImage()
            gii_mag.add_gifti_data_array(gif.GiftiDataArray(mae_mag.astype(np.float32)))
            nib.save(gii_mag, out_mag)

            gii_norm = gif.GiftiImage()
            gii_norm.add_gifti_data_array(gif.GiftiDataArray(mae_norm.astype(np.float32)))
            nib.save(gii_norm, out_norm_path)

            # print(f"已保存: {out_mag}")
            # print(f"已保存: {out_norm_path}")


if __name__ == "__main__":
    # 与 label_efield_reconstruction_gii.py 中 output 结构一致
    root_path_pred = r"/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health"  # 第一组根路径（读取 + 写入 MAE）
    root_path_gt = r"/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health"  # 第二组根路径（仅读取）
    output_relpath_pred = r"visualization/local_efield_vector_final_new_1010_center_based_gt"  # 第一组下的相对路径
    output_relpath_gt = r"visualization/local_sampling_scalp_labels_new_1010_center_efield_vector"  # 第二组下的相对路径（可与第一组不同）

    # 可选：只处理部分 subject
    subjects = ["m2m_A2", "m2m_A9"]
    # subjects = None  # None 表示自动取两组共有的 subject

    compute_and_save_gii_mae(
        root_path_pred,
        root_path_gt,
        output_relpath_pred,
        output_relpath_gt,
        subjects=subjects,
    )
