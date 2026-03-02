# coding=utf-8
import numpy as np
from scipy.spatial import cKDTree


point_target_sub=[-29.89,53.20,36.24]  # m2m_A2左侧DLPFC
# point_target_sub=[44.81,-10.69,39.86]  # m2m_A2右侧M1区
# point_target_sub=[-26.99,-12.45,42.89]  # m2m_A2左侧M1区
sampling_points_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\test_subjects\val_l_sampling_info\m2m_A2\coil_positions.npy'
num_neighbors=10
sampling_points=np.load(sampling_points_path)
tree=cKDTree(sampling_points)
_, neighbors_id = tree.query(point_target_sub, k=num_neighbors)
print(neighbors_id)
sampling_points_neighbors=sampling_points[neighbors_id]
print(sampling_points_neighbors)