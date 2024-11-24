import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_device_fault_data(file_path, num_time_windows):
    # 加载原始数据
    device_data = pd.read_csv(file_path)

    # 打印初始列名和形状
    print(f"初始列名: {device_data.columns.tolist()}")
    print(f"初始数据形状: {device_data.shape}")

    # 只保留列名不为空的列
    device_data = device_data.loc[:, device_data.columns.notnull()]
    device_data = device_data.loc[:, device_data.columns.str.strip() != '']

    # 确保时间戳列存在
    timestamp_column = 'ts'
    if timestamp_column not in device_data.columns:
        raise ValueError(f"数据中缺少时间戳列（{timestamp_column}）。")

    # 转换时间戳为 datetime 格式
    device_data[timestamp_column] = pd.to_datetime(device_data[timestamp_column], errors='coerce')
    device_data = device_data.dropna(subset=[timestamp_column])

    # 转换时间戳为秒数
    device_data['timestamp_seconds'] = (
        device_data[timestamp_column].dt.hour * 3600 +
        device_data[timestamp_column].dt.minute * 60 +
        device_data[timestamp_column].dt.second
    )
    device_data = device_data.drop(columns=[timestamp_column])

    # 打印时间戳处理后的列名和形状
    print(f"时间戳处理后的列名: {device_data.columns.tolist()}")
    print(f"时间戳处理后的数据形状: {device_data.shape}")

    # 将非 0 或 1 的值自动转换为 0
    for col in device_data.columns:
        if col != 'timestamp_seconds':  # 跳过时间戳列
            non_binary_mask = ~device_data[col].isin([0, 1])  # 找出非 0 或 1 的值
            if non_binary_mask.any():
                print(f"警告：列 {col} 包含非 0 或 1 的值，将其转换为 0。")
                device_data.loc[non_binary_mask, col] = 0

    # 打印处理后的列名和形状
    print(f"处理后的列名: {device_data.columns.tolist()}")
    print(f"处理后的数据形状: {device_data.shape}")

    # 提取特征并标准化
    features = device_data.fillna(0)
    print(f"标准化前特征形状: {features.shape}")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"标准化后特征形状: {features_scaled.shape}")

    # 动态检查最终维度
    expected_dim = len(device_data.columns)  # 时间戳列 + 所有故障码列
    if features_scaled.shape[1] != expected_dim:
        raise ValueError(
            f"标准化后特征维度不匹配！实际: {features_scaled.shape[1]}, 预期: {expected_dim} "
            f"（时间戳列 + 故障码列）。"
        )

    # 创建时间特征
    time_steps = np.arange(len(features))
    time_features = np.tile(np.arange(1, num_time_windows + 1), (len(features), 1))

    # 打印时间特征形状
    print(f"时间步数形状: {time_steps[:, None].shape}")
    print(f"时间窗口特征形状: {time_features.shape}")

    # 将时间步数和时间窗口长度作为新的特征
    features_with_time = np.hstack((features_scaled, time_steps[:, None], time_features))
    print(f"组合特征形状: {features_with_time.shape}")

    return torch.tensor(features_with_time, dtype=torch.float32)
