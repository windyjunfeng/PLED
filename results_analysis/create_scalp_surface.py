# coding=utf-8
import numpy as np
from simnibs.utils.mesh_element_properties import ElementTags
from simnibs import mesh_io

mesh = mesh_io.read_msh(r"G:\zhoujunfeng_g\data\cohort_lab_e-field\total\seg\m2m_A1\A1.msh")
fn_geo=r'G:\zhoujunfeng_g\data\cohort_lab_e-field\total\seg\m2m_A1\scalp.geo'
skin_mesh = mesh.crop_mesh(tags = [ElementTags.SCALP_TH_SURFACE])
values=np.array([8,8,8])
values=np.repeat(values[None,:],len(skin_mesh.elm.node_number_list - 1),axis=0)
mesh_io.write_geo_triangles(skin_mesh.elm.node_number_list - 1,skin_mesh.nodes.node_coord, fn_geo,values=values,
                                        name='scalp', mode='ba')