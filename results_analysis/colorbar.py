# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter


def custom_formatter(x, pos):
    # 实际为整数时返回整数字符串；为小数时小数位截止到最后一个非零位（不保留尾部0）
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s


# color_min=0.5
# color_max=1
# n_col=6
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\sampling_points_evaluation_geo\A2\pearson_corr\colorbar.png'
# color_min=0
# color_max=0.8
# n_col=5
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\sampling_points_evaluation_geo\A2\mre\colorbar.png'
# color_min=0
# color_max=1
# n_col=3
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\visualization\m2m_A2\A2_333_45\efield_colorbar.png'
# color_min=0
# color_max=0.2
# n_col=3
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\visualization\m2m_A2\A2_333_45\deviation_colorbar.png'
# color_min=0
# color_max=1.5
# n_col=3
# output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\visualization\m2m_A2\A2_1488_-45\efield_colorbar.png'
color_min=0
color_max=0.5
n_col=3
output_path=r'/data/disk_1/zhoujf/large_data/e-field_dataset/cohort_lab_health/m2m_A2/visualization/efield_mae_colorbar.png'


# fig,ax=plt.subplots(figsize=(6,1))
# fig.subplots_adjust(bottom=0.5)
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')
# norm=mcolors.Normalize(vmin=color_min,vmax=color_max)
# cmap=cm.jet
# sm=cm.ScalarMappable(cmap=cmap,norm=norm)
# colorbar=fig.colorbar(sm,cax=ax, orientation='horizontal')
# tick_positions = np.linspace(color_min, color_max, n_col)  # colorbar上的刻度设置
# colorbar.set_ticks(tick_positions)
# colorbar.ax.set_facecolor('white')
# colorbar.ax.tick_params(colors='black',labelsize=20)
# colorbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
# colorbar.outline.set_edgecolor('black')
# plt.savefig(output_path,dpi=300)


fig,ax=plt.subplots(figsize=(6,1))
fig.subplots_adjust(bottom=0.5)

fig.patch.set_facecolor('black')
ax.set_facecolor('black')

norm=mcolors.Normalize(vmin=color_min,vmax=color_max)
cmap=cm.jet
sm=cm.ScalarMappable(cmap=cmap,norm=norm)
# plt.colorbar(sm,orientation='horizontal')
# colorbar=fig.colorbar(sm,cax=ax, orientation='horizontal')
colorbar=fig.colorbar(sm,cax=ax, orientation='horizontal')
tick_positions = np.linspace(color_min, color_max, n_col)  # colorbar上的刻度设置
colorbar.set_ticks(tick_positions)
# colorbar.ax.tick_params(labelsize=16)
colorbar.ax.set_facecolor('black')
colorbar.ax.tick_params(colors='white',labelsize=36)
colorbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
# matplotlib.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,orientation='horizontal')
# plt.show()
colorbar.outline.set_edgecolor('white')
plt.savefig(output_path,dpi=300)