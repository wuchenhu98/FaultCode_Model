import torch
import pandas as pd
from models.gdn import GDN
from utils.data_processing import load_device_fault_data
from datetime import datetime
import os

# 定义输出文件路径
predictions_dir = 'results/predictions'
anomaly_scores_dir = 'results/anomaly_scores'
correlation_dir = 'results/correlations'
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(anomaly_scores_dir, exist_ok=True)
os.makedirs(correlation_dir, exist_ok=True)

# 加载设备运行数据
num_time_windows = 6
device_data_file = 'data/device_fault_data.csv'
device_data = load_device_fault_data(device_data_file, num_time_windows)

# 检查输入数据形状
if device_data.dim() == 2:  # 如果是二维数据，增加时间维度
    device_data = device_data.unsqueeze(1)  # 添加时间维度

# 模型参数
input_dim = device_data.shape[2]
output_dim = input_dim - 1 - 1 - num_time_windows  # 故障码列数（去掉时间步数和时间窗口特征）
hidden_dim = 64
model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)

# 加载训练好的模型
model.load_state_dict(torch.load('models/gdn_model.pth'))
model.eval()

# 开始预测
print("开始进行预测...")
with torch.no_grad():  # 禁用梯度计算
    predictions, anomaly_scores = model(device_data)

# 展平预测结果和异常得分
predictions_flat = predictions.squeeze(1).detach().numpy()  # 分离计算图并转为 NumPy
anomaly_scores_flat = anomaly_scores[:, :, :output_dim].squeeze(1).detach().numpy()  # 仅保留故障码的异常得分

# 生成列名
original_columns = pd.read_csv(device_data_file).columns.tolist()
fault_columns = original_columns[1:]  # 排除时间戳列，保留故障码列

# 保存预测结果
timestamp_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
prediction_file = os.path.join(predictions_dir, f'predictions_{timestamp_str}.csv')
anomaly_file = os.path.join(anomaly_scores_dir, f'anomaly_scores_{timestamp_str}.csv')
correlation_file = os.path.join(correlation_dir, f'correlations_{timestamp_str}.csv')

# 保存预测结果为 DataFrame
predictions_df = pd.DataFrame(predictions_flat, columns=fault_columns)
predictions_df.to_csv(prediction_file, index=False)

# 保存异常得分为 DataFrame
anomaly_scores_df = pd.DataFrame(anomaly_scores_flat, columns=fault_columns)
anomaly_scores_df.to_csv(anomaly_file, index=False)

# 计算故障码之间的相关性
print("计算故障码之间的相关性...")
correlation_matrix = anomaly_scores_df.corr()

# 提取关联度较高的故障码对
high_correlation_threshold = 0.5  # 设定相关性阈值
high_correlations = []
for i, col in enumerate(correlation_matrix.columns):
    for j in range(i + 1, len(correlation_matrix.columns)):
        correlation_value = correlation_matrix.iloc[i, j]
        if correlation_value >= high_correlation_threshold:
            high_correlations.append(
                (col, correlation_matrix.columns[j], round(correlation_value, 4))  # 保留四位小数
            )

# 检查是否有高关联故障码对
if high_correlations:
    high_correlations_df = pd.DataFrame(high_correlations, columns=['故障码1', '故障码2', '关联度系数'])
    high_correlations_df.to_csv(correlation_file, index=False)
    print(f"关联度较高的故障码已保存到 {correlation_file}")
else:
    print("未找到高关联度的故障码对，调整阈值或检查数据质量。")

# 保存完整的相关性矩阵供调试
correlation_matrix_file = os.path.join(correlation_dir, f'correlation_matrix_{timestamp_str}.csv')
correlation_matrix.to_csv(correlation_matrix_file)
print(f"完整相关性矩阵已保存到 {correlation_matrix_file}")
