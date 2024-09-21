import pandas as pd

# 读取Parquet文件
parquet_file_path = 'compress_data/parquet/city0-4G-1M-compress.parquet'
df = pd.read_parquet(parquet_file_path)

# 将DataFrame转换为CSV并保存
csv_file_path = 'test.csv'
df.to_csv(csv_file_path, index=False)