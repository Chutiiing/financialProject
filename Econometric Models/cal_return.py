import pandas as pd
import numpy as np
import os

# 原始文件夹路径
input_folder = r'D:\xmu\毕设\data\source'
# 新文件夹路径
output_folder = r'D:\xmu\毕设\data\return'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

i = 1
# 遍历原始文件夹中的所有csv文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 读取csv文件
        df = pd.read_csv(os.path.join(input_folder, filename))
        
        # 计算收益率，考虑昨日收盘价可能为0的情况
        changes = df.iloc[:, 2].pct_change()  # 计算变化率
        # 使用np.where处理昨日收盘价为0的情况
        df['收益率'] = np.where(df.iloc[:, 2].shift(1) == 0, 0, changes)
        df['收益率'].fillna(0, inplace=True)  # 将NaN值（通常是第一行）替换为0
        
        # 在第七列之后插入两列，第八列为空，第九列为收益率
        df.insert(7, '空列', '')
        df.insert(8, '收益率', df.pop('收益率'))
        
        # 保存新的csv文件到输出文件夹
        df.to_csv(os.path.join(output_folder, filename), index=False)
        print(i)
        i += 1

print("处理完成，所有文件已保存至新的文件夹。")



