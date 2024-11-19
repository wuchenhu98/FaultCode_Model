import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_device_data(file_path, num_time_windows):
    # 读取设备运行数据
    device_data = pd.read_csv(file_path)

    # 将时间戳列 (_ts) 转换为数值特征（以秒为单位）
    device_data['_ts'] = pd.to_datetime(device_data['_ts'], errors='coerce')
    device_data['timestamp_seconds'] = (
        device_data['_ts'].dt.hour * 3600 +
        device_data['_ts'].dt.minute * 60 +
        device_data['_ts'].dt.second
    )
    
    # 故障码特征列
    fault_columns = device_data.columns[1:-1]  # 从第2列到倒数第2列
    device_data[fault_columns] = device_data[fault_columns].fillna(0).astype(int)  # 确保为整数

    # 提取所有特征（时间戳 + 故障码特征）
    features = device_data[["timestamp_seconds"] + list(fault_columns)]

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 创建时间特征
    time_steps = np.arange(len(features))
    time_features = np.tile(np.arange(1, num_time_windows + 1), (len(features), 1))

    # 将时间步数和时间窗口长度作为新的特征
    features_with_time = np.hstack((features_scaled, time_steps[:, None], time_features))

    return torch.tensor(features_with_time, dtype=torch.float32)

def load_maintenance_data(file_path):
    # 加载维修日志数据
    maintenance_data = pd.read_csv(file_path)
    maintenance_data_onehot = pd.get_dummies(maintenance_data["repair_type"])
    return maintenance_data_onehot, maintenance_data_onehot.columns.tolist()

def generate_time_window_labels(maintenance_data_df, num_time_windows):
    # 生成未来时间窗口的标签
    time_window_labels = []
    for i in range(1, num_time_windows + 1):
        window_label = maintenance_data_df.shift(-i).fillna(0).astype(float)
        time_window_labels.append(window_label)
    time_window_labels = [torch.tensor(label.values, dtype=torch.float32) for label in time_window_labels]
    
    return torch.stack(time_window_labels, dim=1)
