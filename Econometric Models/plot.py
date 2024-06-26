import pandas as pd
import matplotlib.pyplot as plt

# 使用GBK编码来读取数据
df = pd.read_csv(r'D:\xmu\毕设\data\garch\300_000001_data_2010_2023.csv', encoding='gbk')

# 选择最后365天的数据
df_last_365 = df.tail(365)

# 创建图表对象
fig, ax1 = plt.subplots()

# 绘制条件方差（左边纵轴）
color = 'tab:red'
ax1.set_ylabel('Conditional Standard Deviation', color=color)
ax1.plot(df_last_365['条件标准差'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 隐藏x轴的所有元素
ax1.xaxis.set_visible(False)

# 创建一个共享x轴的新轴对象（右边纵轴）
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Returns', color=color)
ax2.plot(df_last_365['收益率'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 设置图表标题和布局
plt.title('Conditional Standard Deviation and Returns Over the Last 365 Days')
fig.tight_layout()

# 显示图表
plt.show()




