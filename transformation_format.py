import os
import pandas as pd

def convert_parquet_to_csv(directory):
    # 获取文件夹名称
    folder_name = os.path.basename(os.path.normpath(directory))
    

    # 创建 'data/' 文件夹（如果不存在）
    output_dir = os.path.join('data')
    os.makedirs(output_dir, exist_ok=True)

    # 列出文件夹下所有.parquet文件
    for file in os.listdir(directory):
        if file.endswith('.parquet'):
            # 获取文件名（不带扩展名）
            file_name = os.path.splitext(file)[0]

            # 假设文件名中以 '_' 分隔的子串结构，提取某个子串
            # 例如：'data_2024_info.parquet' -> 提取 '2024'
            parts = file_name.split('_')  # 根据需要选择分隔符
            if len(parts) > 1:
                extracted_part = parts[4]  # 提取第4个子串
            else:
                extracted_part = file_name  # 如果没有分隔符，使用整个文件名

            # 构造完整的parquet文件路径F
            parquet_file_path = os.path.join(directory, file)

            # 读取parquet文件
            df = pd.read_parquet(parquet_file_path)

            # 使用提取的子串命名新的CSV文件
            csv_file_path = os.path.join(output_dir, f"{folder_name}-{extracted_part}.csv")

            # 将DataFrame保存为CSV文件
            df.to_csv(csv_file_path, index=False)

         

# 示例用法
convert_parquet_to_csv('orignal_data/city0-4G-1M')
