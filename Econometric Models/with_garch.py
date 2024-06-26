import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from arch import arch_model

# 数据加载和预处理
def load_and_preprocess_data(filepath):
    # 加载数据
    data = pd.read_csv(filepath)
    # 将日期列转换为datetime类型
    data['日期'] = pd.to_datetime(data['日期'])
    # 设定日期为索引
    data.set_index('日期', inplace=True)
    return data

# 特征工程
def feature_engineering(data, volatilities, history_days=50):
    features = []
    labels = []
 
    for i in range(history_days, len(data)):
        # 提取主要特征
        feature_slice = data.iloc[i-history_days:i, 1:7].values
        # 提取相应的条件标准差，并进行reshape以与其他特征在维度上保持一致
        volatility_slice = volatilities[i-history_days:i].reshape(-1, 1)
        # 合并特征和条件标准差
        combined_features = np.hstack((feature_slice, volatility_slice))
        features.append(combined_features)
        labels.append(data.iloc[i, -1])  # 收益率列
 
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

# 定义一个函数来拟合GARCH模型并返回条件标准差
def estimate_garch_volatilities(data):
    data = data * 100  # 将数据放大100倍，通常用于处理非常小的数值
    
    # 使用GARCH(1,1)作为初始模型
    model = arch_model(data, vol='Garch', p=1, q=1)
    
    # 拟合模型
    fitted_model = model.fit(disp='off')  # disp='off'关闭拟合过程中的输出信息

    # 输出BIC，并尝试不同的阶数以找到最佳模型
    best_bic = fitted_model.bic
    best_order = (1, 1)
    best_model = fitted_model
    
    # 探索不同的阶数组合
    for p in range(1, 10):
        for q in range(0, 10):
            model = arch_model(data, vol='Garch', p=p, q=q)
            res = model.fit(disp='off')
            if res.bic < best_bic:
                best_bic = res.bic
                best_order = (p, q)
                best_model = res
    
    # 获取拟合模型的条件标准差
    conditional_volatility = np.array(best_model.conditional_volatility.tolist()) / 100
    
    # 预测未来5天的条件标准差
    forecast = best_model.forecast(horizon=5)
    future_volatility = forecast.variance.values[-1, :]**0.5 / 100
    
    # 合并当前的条件标准差与未来5天的预测
    extended_volatility = np.concatenate([conditional_volatility, future_volatility])
    
    return conditional_volatility, extended_volatility


# 模型构建
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

#离线阶段
def train_model(model, X_train, y_train, X_valid, y_valid, epochs=200, batch_size=32):
    # 复制一个新模型用于验证
    validation_model = clone_model(model)
    validation_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # 最佳损失初始化为无穷大，用于早停判断
    best_loss = float('inf')
    patience_counter = 0
    patience = 8  # 早停耐心值
    last_val_loss=float('inf')
    val_loss=float('inf')
    for epoch in range(epochs):
        # 在训练集上训练模型
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

        # 拷贝当前模型的权重到验证模型
        validation_model.set_weights(model.get_weights())

        # 在验证集上进行增量训练
        validation_model.fit(X_valid, y_valid, batch_size=batch_size, epochs=5, verbose=0)
        
        last_val_loss=val_loss
        # 评估验证集上的损失
        val_loss = validation_model.evaluate(X_valid, y_valid, verbose=0)

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

        # 检查是否达到早停条件
        if val_loss < last_val_loss:
            patience_counter = 0  # 重置耐心计数器
            # 保存当前最佳模型
            model.save('best_model_garch.h5')
        else:
            patience_counter += 1  # 增加耐心计数器

        if patience_counter >= patience:
              
            print("Early stopping...")
            break

    # 加载最佳模型
    model = tf.keras.models.load_model('best_model_garch.h5')
    return model


# 在线训练和更新模型
def test_model(model, X_valid, y_valid, X_test, y_test, y_train, batch_size=32):
    incremental_model = clone_model(model)
    incremental_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    incremental_model.set_weights(model.get_weights())

    predictions = []
    history_X = np.copy(X_valid)
    history_y = np.copy(y_valid)
    full_history_y = np.concatenate([y_train, y_valid])  # 包括到验证数据为止的历史数据

    # 循环处理每个5天的测试数据，确保每一轮开始前数据是最新的
    for start in range(0, len(X_test)-50, 5):
        end = min(start + 5, len(X_test)-50)
        increment_length = end - start

        # 更新数据集以重新估计条件标准差
        updated_data = np.concatenate([full_history_y, y_test[:start]])  # 注意这里包括了当前批次的真实数据

        # 重新估计GARCH模型并更新条件标准差
        _, new_extended_vol = estimate_garch_volatilities(updated_data)

        # 更新历史数据中的条件标准差信息
        if len(new_extended_vol) > len(history_y):
            # 生成新的历史特征矩阵，将历史X中的条件标准差部分替换为新的标准差
            history_X[:, :, -1] = new_extended_vol[:len(history_y)].reshape(-1, 1)  # 假设条件标准差是最后一列

        # 准备当前批次的特征（包括新的条件标准差）
        current_features,_ = feature_engineering(X_test[start:end+50],new_extended_vol[-increment_length-50:])
    
        # 用当前历史数据和特征进行增量学习
        incremental_model.fit(history_X, history_y, batch_size=batch_size, epochs=50, verbose=0)

        # 进行预测
        prediction = incremental_model.predict(current_features, batch_size=batch_size)
        predictions.extend(prediction.flatten())

        # 更新历史数据以包括这个测试周期的数据
        history_X = np.vstack([history_X, current_features])
        history_y = np.concatenate([history_y, y_test[start:end]])
        full_history_y = np.concatenate([full_history_y, y_test[start:end]])
        print(start)
    predictions = np.array(predictions)
    return calculate_performance_metrics(y_test[:len(predictions)], predictions)



def calculate_performance_metrics(y_true, y_pred, window_size=100, step_size=100):
    # 将numpy数组转换为pandas的Series
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)

    # 计算整体IC和RankIC
    overall_ic = np.corrcoef(y_true_series, y_pred_series)[0, 1]
    overall_rank_ic = np.corrcoef(y_true_series.rank(), y_pred_series.rank())[0, 1]

    # 滚动窗口收集IC和RankIC
    ics = []
    rank_ics = []

    # 确保有足够的数据进行至少一个窗口的计算
    if len(y_true_series) >= window_size:
        for start in range(0, len(y_true_series) - window_size + 1, step_size):
            end = start + window_size
            window_true = y_true_series[start:end]
            window_pred = y_pred_series[start:end]

            # 计算当前窗口的IC
            ic = np.corrcoef(window_true, window_pred)[0, 1]
            ics.append(ic)

            # 计算当前窗口的RankIC
            rank_ic = np.corrcoef(window_true.rank(), window_pred.rank())[0, 1]
            rank_ics.append(rank_ic)

    # 计算ICIR和RankICIR
    if len(ics) > 0 and len(rank_ics) > 0:
        ic_std = np.std(ics)
        rank_ic_std = np.std(rank_ics)

        if ic_std != 0:
            icir = overall_ic / ic_std

        if rank_ic_std != 0:
            rank_icir = overall_rank_ic / rank_ic_std

    else:
        print("Not enough data to calculate window metrics or standard deviations are zero.")
    return overall_ic,overall_rank_ic,icir,rank_icir

# 主程序
# 主程序
def main():
    directory = './data500'  # 假设数据文件夹在脚本同级目录下
    metrics = []
    i=1
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            data = load_and_preprocess_data(filepath)
            GARCH_offline,_ = estimate_garch_volatilities(data.iloc[:2300, -1].values)
            features, labels = feature_engineering(data.iloc[:2300], GARCH_offline)
            X_train = features[:1950]
            X_valid = features[1950:2250]
            y_train = labels[:1950]
            y_valid = labels[1950:2250]
            X_test, y_test = data.iloc[2200:, 0:7], data.iloc[2250:, -1]
            model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            model = train_model(model, X_train, y_train, X_valid, y_valid)
            ic, rank_ic, icir, rank_icir = test_model(model, X_valid, y_valid, X_test, y_test, y_train)
            metrics.append((ic, rank_ic, icir, rank_icir))
            print(i)
            i=i+1
    ic_avg = np.mean([m[0] for m in metrics])
    rank_ic_avg = np.mean([m[1] for m in metrics])
    icir_avg = np.mean([m[2] for m in metrics])
    rank_icir_avg = np.mean([m[3] for m in metrics])
    print("Average IC:", ic_avg)
    print("Average RankIC:", rank_ic_avg)
    print("Average ICIR:", icir_avg)
    print("Average RankICIR:", rank_icir_avg)

if __name__ == "__main__":
    main()