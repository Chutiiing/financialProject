import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('p_values_summary.csv')  # 替换 'your_file.csv' 为你的文件名

# 提取第一列和第二列数据
x = data.iloc[:, 1]  # 第二列数据
y = data.iloc[:, 2]  # 第三列数据
column_x_name = data.columns[1]  # 第二列的列名
column_y_name = data.columns[2]  # 第三列的列名

# 创建图像和坐标轴
fig, ax = plt.subplots()

# 绘制散点图
sc = ax.scatter(x, y, c=((x < 0.025) & (y < 0.025)), cmap='coolwarm', edgecolor='none')

# 设置坐标轴
ax.set_xlabel(column_x_name)
ax.set_ylabel(column_y_name)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 创建一个颜色条，红色表示两个特征都小于0.025的点
cbar = plt.colorbar(sc)
cbar.set_label('Red if both < 0.025')

# 显示图像
plt.show()
