import pandas as pd
import random
from datetime import datetime, timedelta

# 生成随机递增时间戳数据（严格递增且无重复）
def generate_strictly_increasing_timestamps(start_time, num_records, min_increment_seconds=1, max_increment_seconds=15):
    timestamps = []
    current_time = start_time
    for _ in range(num_records):
        timestamps.append(current_time)
        # 确保每次递增至少 min_increment_seconds，避免重复时间戳
        increment = random.randint(min_increment_seconds, max_increment_seconds)
        current_time += timedelta(seconds=increment)
    return timestamps

# 配置参数
start_time = datetime(2024, 1, 1, 0, 0, 0)  # 起始时间
num_records = 3000  # 记录数量
min_increment_seconds = 1  # 最小递增秒数
max_increment_seconds = 15  # 最大递增秒数

# 生成时间戳
timestamps = generate_strictly_increasing_timestamps(start_time, num_records, min_increment_seconds, max_increment_seconds)

# 创建DataFrame
df = pd.DataFrame({'ts': timestamps})

# 保存为CSV文件
output_file = 'strictly_increasing_timestamps_3000.csv'
df.to_csv(output_file, index=False)

print(f"CSV文件已生成: {output_file}")
