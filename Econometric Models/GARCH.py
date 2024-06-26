import pandas as pd
import os
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

# 设置文件夹路径
input_folder = r'D:\xmu\毕设\data\garch_passed'
output_folder = r'D:\xmu\毕设\data\garch'
os.makedirs(output_folder, exist_ok=True)

# 初始化一个DataFrame来存储GARCH模型阶数
orders_df = pd.DataFrame(columns=['Filename', 'p', 'q'])

i=1
# 遍历所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        
        # 确保收益率列存在
        if '收益率' in df.columns and df.columns[7] == '空列':  # 假设第八列标签为'空列'
            returns = df['收益率']*100

            # 选择最佳GARCH模型
            best_bic = float('inf')
            best_model = None
            best_p = None
            best_q = None
            for p in range(1, 10):
                for q in range(0, 10):
                    if p!=0 or q!=0:
                        model = arch_model(returns, vol='Garch', p=p, q=q)
                        try:
                            res = model.fit(disp='off', show_warning=False)
                            if res.bic < best_bic:
                                best_bic = res.bic
                                best_model = res
                                best_p = p
                                best_q = q
                        except ValueError:
                        # 捕获模型无法拟合的情况
                            continue

            # 如果找到最佳模型，计算条件标准差并保存
            if best_model is not None:
                df = df.rename(columns={df.columns[7]: "条件标准差"})
                df.iloc[:, 7] = (best_model.conditional_volatility)/10000  # 填充第八列（索引7）
                new_file_path = os.path.join(output_folder, filename)
                df.to_csv(new_file_path, index=False)
                temp_df = pd.DataFrame({'Filename': [filename], 'p': [best_p], 'q': [best_q]})
                orders_df = pd.concat([orders_df, temp_df], ignore_index=True)
    print(i)
    i=i+1
# 保存阶数到CSV文件
orders_csv_path = os.path.join(output_folder, 'garch_orders.csv')
orders_df.to_csv(orders_csv_path, index=False)

print("处理完成，所有文件及GARCH模型阶数已保存至新的文件夹。")
