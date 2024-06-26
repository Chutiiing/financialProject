import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

# 设置原始文件夹和输出文件夹的路径
input_folder = r'D:\xmu\毕设\data\return'
output_folder = r'D:\xmu\毕设\data\garch_passed'
# 创建输出文件夹（如果不存在的话）
os.makedirs(output_folder, exist_ok=True)

i=1
# 遍历原始文件夹中的所有csv文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # 确保收益率列存在
        if '收益率' in df.columns:
            returns = df['收益率']

            # ADF 检验
            adf_p_value = adfuller(returns.dropna())[1]  # ADF检验前需要剔除NaN，因为ADF不处理NaN

            # ARCH效应检验
            arch_p_value = het_arch(returns.dropna())[1]  # ARCH检验也需要剔除NaN

            # 检查两个检验的p值是否都小于0.05
            if adf_p_value < 0.05 and arch_p_value < 0.05:
                # 如果通过检验，保存文件到新的文件夹
                df.to_csv(os.path.join(output_folder, filename), index=False)
    print(i)
    i=i+1
print("检验完成，符合条件的文件已保存至新的文件夹。")
