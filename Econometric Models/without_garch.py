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
def feature_engineering(data, history_days=50):
    features = []
    labels = []
 
    for i in range(history_days, len(data)):
        feature_slice = data.iloc[i-history_days:i, 1:7].values
        features.append(feature_slice)  # 不再使用 flatten()
        labels.append(data.iloc[i, -1])  # 收益率列
 
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

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
            # 保存当前最佳模型
            model.save('best_model.h5')
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1  # 增加耐心计数器

        if patience_counter >= patience:              
            print("Early stopping...")
            break

    # 加载最佳模型
    model = tf.keras.models.load_model('best_model.h5')
    return model
# 在线训练和更新模型

def test_model(model, X_valid, y_valid, X_test, y_test, batch_size=32):
    # 模型用于预测的副本
    incremental_model = clone_model(model)
    incremental_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    incremental_model.set_weights(model.get_weights())

    predictions = []

    # 初始化历史数据为验证集数据
    history_X = X_valid.copy()
    history_y = y_valid.copy()

    # 按照5天增量进行测试
    for start in range(0, len(X_test), 5):
        end = start + 5
        if end > len(X_test):
            end = len(X_test)
        incremental_model = clone_model(model)
        incremental_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        incremental_model.set_weights(model.get_weights())
        X_increment = X_test[start:end]
        y_increment = y_test[start:end]

        # 使用所有已知历史数据进行增量学习
        incremental_model.fit(history_X, history_y, batch_size=batch_size, epochs=5, verbose=0)

        # 进行预测
        prediction = incremental_model.predict(X_increment, batch_size=batch_size)
        predictions.extend(prediction.flatten())

        # 更新历史数据以包括这个测试周期的数据
        history_X = np.vstack([history_X, X_increment])
        history_y = np.concatenate([history_y, y_increment])
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

def process_stock_data(filepath):
    data = load_and_preprocess_data(filepath)
    features, labels = feature_engineering(data)
    X_train, y_train = features[:1950], labels[:1950]
    X_valid, y_valid = features[1950:2250], labels[1950:2250]
    X_test, y_test = features[2250:], labels[2250:]
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model = train_model(model, X_train, y_train, X_valid, y_valid)
    ic, rank_ic, icir, rank_icir = test_model(model, X_valid, y_valid, X_test, y_test)
    return ic, rank_ic, icir, rank_icir

def main():
    directory = r'./data500'
    metrics = []
    i=1
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            metrics.append(process_stock_data(filepath))
        print(i)
        i=i+1
    # 计算指标平均值
    ic_avg = np.mean([metric[0] for metric in metrics])
    rank_ic_avg = np.mean([metric[1] for metric in metrics])
    icir_avg = np.mean([metric[2] for metric in metrics])
    rank_icir_avg = np.mean([metric[3] for metric in metrics])
    
    print("Average IC:", ic_avg)
    print("Average RankIC:", rank_ic_avg)
    print("Average ICIR:", icir_avg)
    print("Average RankICIR:", rank_icir_avg)

if __name__ == "__main__":
    main()