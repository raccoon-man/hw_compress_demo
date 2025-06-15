import os
import pandas as pd


def convert_parquet_to_csv(directory):
    # 获取文件夹名称
    folder_name = os.path.basename(os.path.normpath(directory))
    combined_dfs = {}
    parquet_size_sum = {}

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

            # 构造完整的parquet文件路径
            parquet_file_path = os.path.join(directory, file)

            # 读取parquet文件
            df = pd.read_parquet(parquet_file_path)
            file_size = os.path.getsize(parquet_file_path)
            if 'gridid' in df.columns:
                df = df.drop(['gridid'], axis=1)

            # 拼接相同 extracted_part 的 DataFrame
            if extracted_part in combined_dfs:
                combined_dfs[extracted_part] = pd.concat(
                    [combined_dfs[extracted_part], df], ignore_index=True)
                parquet_size_sum[extracted_part] += file_size
            else:
                combined_dfs[extracted_part] = df
                parquet_size_sum[extracted_part] = file_size

    # 将合并后的 DataFrame 保存为 CSV 文件
    for extracted_part, df in combined_dfs.items():
        if len(df) < 2:
            continue
        csv_file_path = os.path.join(output_dir, f"{extracted_part}.csv")
        df.to_csv(csv_file_path, index=False)

    # 保存统计信息到 CSV
    data = {'File Name': list(parquet_size_sum.keys()),
            'Parquet Size': list(parquet_size_sum.values())}
    df = pd.DataFrame(data)
    df.to_csv('parquet_size.csv', index=False)


convert_parquet_to_csv('original_data/')
    