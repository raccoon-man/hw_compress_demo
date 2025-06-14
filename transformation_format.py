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
            
            # 提取站点名和ID
            parts = file_name.split('.')
            if len(parts) < 2:
                print(f"跳过格式异常的文件: {file}")
                continue
                
            site_name = parts[0]
            
            # 提取ID (假设ID在第5个下划线分隔的部分，索引为4)
            id_parts = parts[1].split('_')
            if len(id_parts) < 5:
                print(f"跳过格式异常的文件: {file}")
                continue
                
            id_value = id_parts[4]  # 索引4对应第5个部分
            
            # 生成组合键
            combined_key = f"{site_name}_{id_value}"

            # 构造完整的parquet文件路径
            parquet_file_path = os.path.join(directory, file)

            # 读取parquet文件
            df = pd.read_parquet(parquet_file_path)
            file_size = os.path.getsize(parquet_file_path)
            if 'gridid' in df.columns:
                df = df.drop(['gridid'], axis=1)

            # 拼接相同站点和ID的 DataFrame
            if combined_key in combined_dfs:
                combined_dfs[combined_key] = pd.concat(
                    [combined_dfs[combined_key], df], ignore_index=True)
                parquet_size_sum[combined_key] += file_size
            else:
                combined_dfs[combined_key] = df
                parquet_size_sum[combined_key] = file_size

    # 将合并后的 DataFrame 保存为 CSV 文件（追加模式）
    for combined_key, df in combined_dfs.items():
        if len(df) < 2:
            continue
            
        csv_file_path = os.path.join(output_dir, f"{combined_key}.csv")
        
        # 检查文件是否已存在
        file_exists = os.path.exists(csv_file_path)
        
        # 以追加模式写入CSV，存在则不写表头，不存在则写表头
        df.to_csv(csv_file_path, mode='a', index=False, header=not file_exists)
        
        if file_exists:
            print(f"已追加数据到文件: {csv_file_path}")
        else:
            print(f"已创建新文件: {csv_file_path}")

    # 保存统计信息到 CSV
    data = {'File Name': list(parquet_size_sum.keys()),
            'Parquet Size': list(parquet_size_sum.values())}
    df = pd.DataFrame(data)
    df.to_csv('parquet_size.csv', index=False)
    print("已保存统计信息文件: parquet_size.csv")


convert_parquet_to_csv('original_data/')