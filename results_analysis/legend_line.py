# coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D


output_path=r'/data/disk_2/zhoujunfeng/code/efield_calculation/outputs/evaluation/cohort_lab_health/results_analysis/efield_angleres_1_sampling_direction_evaluation_point_plot/total/legend.png'
# 定义 colormap 和归一化器
cmap = cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=1)

# 创建一个颜色映射器
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

# 定义图例的标签、对应的数值、线型和 marker
# labels_values_linestyles_markers = [
#     ('Origin', 0.25, 'dotted', 's'),  # 点和圆圈标记
#     ('Baseline', 0.5, '--', '^'), # 虚线和三角形标记
#     ('PLED (Ours)', 0.75, '-', '*')  # 实线和方形标记
# ]
labels_values_linestyles = [
    ('Origin', 0.25, 'dotted'),  # 点
    ('Baseline', 0.5, '--'), # 虚线
    ('PLED (Ours)', 0.75, '-')  # 实线
]

# 创建一个 figure 对象
fig = plt.figure(figsize=(6, 2))

# 创建一个轴，用于显示图例
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_axis_off()

# 创建图例的元素
# legend_elements = [
#     Line2D([0], [0], color=sm.to_rgba(value), lw=2, linestyle=linestyle, label=label, marker=marker, markersize=10)
#     for label, value, linestyle, marker in labels_values_linestyles_markers
# ]
legend_elements = [
    Line2D([0], [0], color=sm.to_rgba(value), lw=2, linestyle=linestyle, label=label)
    for label, value, linestyle in labels_values_linestyles
]

# 创建图例
legend = ax.legend(handles=legend_elements, loc='center', frameon=False, handlelength=3, handleheight=2, fontsize=14)

# 调整图例中行与行之间的间距
legend._legend_box.sep = 10

# 保存图例为图片
plt.savefig(output_path, bbox_inches='tight', dpi=300)

# 显示图形
# plt.show()