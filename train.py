import torch
import torch.optim as optim
import os
import pandas as pd
from models.gdn import GDN
from utils.data_processing import load_device_fault_data

# 文件路径
device_data_file = 'data/device_fault_data.csv'

# 加载数据
num_time_windows = 6
device_data = load_device_fault_data(device_data_file, num_time_windows)

# 检查输入数据形状
if device_data.dim() == 2:  # 如果是二维数据，增加时间维度
    device_data = device_data.unsqueeze(1)  # 添加时间维度

# 加载原始数据并过滤无效列
raw_data = pd.read_csv(device_data_file)

# 只保留列名不为空的列
raw_data = raw_data.loc[:, raw_data.columns.notnull()]  # 保留列名不为空的列
raw_data = raw_data.loc[:, raw_data.columns.str.strip() != '']  # 进一步检查非空字符串的列

# 动态获取故障码列
fault_code_columns = raw_data.columns[1:]  # 第一列是时间戳，其余是故障码列
output_dim = len(fault_code_columns)  # 更新故障码数量

# 确保训练数据和标签形状一致
expected_dim = output_dim + 1 + 1 + num_time_windows  # 时间戳列 + 故障码列 + 时间步数 + 时间窗口特征
if device_data.shape[2] != expected_dim:
    raise ValueError(
        f"输入数据维度不匹配：device_data.shape[2] = {device_data.shape[2]}, "
        f"而预期维度 = {expected_dim}（时间戳列 + 故障码列 + 时间步数 + 时间窗口特征）。"
    )

# 数据划分
train_size = int(0.8 * device_data.shape[0])
train_data = device_data[:train_size]  # 前80%作为训练集
test_data = device_data[train_size:]  # 后20%作为测试集

# 模型参数
input_dim = device_data.shape[2]  # 输入维度
hidden_dim = 64  # 隐藏层维度
model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
print("开始训练模型...")
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 获取模型输出
    predictions, anomaly_scores = model(train_data)

    # 提取故障码列作为标签
    labels = train_data[:, :, 1:1+output_dim]  # 排除时间戳列和额外特征，仅使用故障码列

    # 确保 predictions 和 labels 的形状一致
    if predictions.shape != labels.shape:
        raise ValueError(
            f"预测输出形状 {predictions.shape} 与标签形状 {labels.shape} 不一致！"
        )

    # 计算损失
    loss = torch.nn.functional.mse_loss(predictions, labels)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 保存模型
save_path = 'models/gdn_model.pth'
if os.path.exists(save_path):
    os.remove(save_path)
torch.save(model.state_dict(), save_path)
print(f"模型已保存到 {save_path}")
