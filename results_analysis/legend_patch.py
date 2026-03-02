# coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch


output_path=r'G:\zhoujunfeng_g\code\deep_learning_e-field_large_files\evaluation\results_analysis\box_plot_viridis_nolegend\legend.png'
# 定义 colormap 和归一化器
cmap = cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=1)

# 创建一个颜色映射器
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

# 创建一个 figure 对象
fig = plt.figure(figsize=(6, 2))

# 创建一个轴，用于显示图例
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_axis_off()

# 创建图例的元素
legend_elements = [
    Patch(facecolor=sm.to_rgba(0.25), edgecolor='black', label='Origin',linewidth=2),
    Patch(facecolor=sm.to_rgba(0.5), edgecolor='black', label='Baseline',linewidth=2),
    Patch(facecolor=sm.to_rgba(0.75), edgecolor='black', label='PLED (Ours)',linewidth=2),
]

# 创建图例
legend = ax.legend(handles=legend_elements, loc='center', frameon=False, handlelength=3, handleheight=2, fontsize=14,ncol=3)

# 保存图例为图片
plt.savefig(output_path, bbox_inches='tight', dpi=300)

# 显示图形
# plt.show()