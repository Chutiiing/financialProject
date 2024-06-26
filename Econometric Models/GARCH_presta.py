import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

# 原始文件夹路径
input_folder = r'D:\xmu\毕设\data\return'
# 输出文件夹路径
output_folder = r'D:\xmu\毕设\data\garch_examine'
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 初始化一个DataFrame来存储所有的p值结果
p_values_df = pd.DataFrame(columns=['Filename', 'ADF_p_value', 'ARCH_p_value'])

i=1
# 遍历原始文件夹中的所有csv文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 读取csv文件
        df = pd.read_csv(os.path.join(input_folder, filename))
        # 检测收益率列是否存在，若不存在，则跳过
        if '收益率' not in df.columns:
            continue
        # 获取收益率列数据
        returns = df['收益率'].dropna()  # 确保没有空值
        
        # ADF检验，用于检查数据的平稳性
        adf_result = adfuller(returns)
        adf_p_value = adf_result[1]  # ADF检验的p值
        
        # Engle的ARCH效应检验，用于检查残差中的条件异方差
        _, arch_p_value, _, _ = het_arch(returns)
        
        # 创建一个临时DataFrame来存储当前文件的结果
        temp_df = pd.DataFrame({'Filename': [filename],
                                'ADF_p_value': [adf_p_value],
                                'ARCH_p_value': [arch_p_value]})

        # 使用concat合并DataFrame
        p_values_df = pd.concat([p_values_df, temp_df], ignore_index=True)
    print(i)
    i=i+1
# 保存汇总的p值表格到新的csv文件中
p_values_df.to_csv(os.path.join(output_folder, 'p_values_summary.csv'), index=False)

print("处理完成，所有检验的p值已保存至新的文件夹。")
