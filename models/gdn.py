import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_time_windows):
        super(GDN, self).__init__()
        self.num_time_windows = num_time_windows

        # 时间序列嵌入层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)  # 时间序列处理的GRU
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 用于存储每个特征的历史均值和标准差
        self.feature_means = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.feature_stds = nn.Parameter(torch.ones(input_dim), requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, num_time_steps, input_dim]
        
        Returns:
            predictions: 故障预测结果，形状为 [batch_size, num_time_steps, output_dim]
            anomaly_scores: 异常得分，形状为 [batch_size, num_time_steps, input_dim]
        """
        # 计算每个特征与均值的距离，并归一化到标准差
        normalized_distances = torch.abs((x - self.feature_means) / (self.feature_stds + 1e-6))
        anomaly_scores = torch.sigmoid(normalized_distances)  # 将距离映射到 [0, 1] 作为异常得分

        # 时间序列处理
        batch_size, num_time_steps, input_dim = x.shape
        x = x.view(-1, input_dim)  # 展平时间步，形状变为 [batch_size * num_time_steps, input_dim]
        x = F.relu(self.fc1(x))  # 嵌入层
        x = x.view(batch_size, num_time_steps, -1)  # 恢复时间步维度
        x, _ = self.gru(x)  # GRU输出，形状为 [batch_size, num_time_steps, hidden_dim]

        # 最终预测
        predictions = torch.sigmoid(self.fc2(x))  # 将输出通过 sigmoid 转换为概率

        return predictions, anomaly_scores

    def update_statistics(self, data):
        """
        更新输入数据的均值和标准差，用于异常检测。
        
        Args:
            data: 输入张量，形状为 [batch_size, num_time_steps, input_dim]
        """
        with torch.no_grad():
            self.feature_means.data = data.mean(dim=(0, 1))  # 计算所有时间步和批次的均值
            self.feature_stds.data = data.std(dim=(0, 1)) + 1e-6  # 避免除以0
