import pandas as pd
import numpy as np

def round_with_1_percent_error(value):
    """对整数值进行四舍五入，使低数位归零，误差不超过1%"""
    if pd.isna(value):  # 跳过 NaN
        return value
    magnitude = int(np.floor(np.log10(abs(value)) - 2)) if value != 0 else 0  # 计算保留数位
    return round(value, -magnitude)

def process_csv_with_rounding(data, exclude_column):
    """读取 CSV，对整数列（不包括最后一列和指定列）进行低数位归零处理，并导出新的 CSV，确保数据类型不变"""
    # 读取整个 CSV，同时保留原始数据类型
    df = data
    original_dtypes = df.dtypes.copy()
    
    # 获取所有整数列，但排除最后一列和指定列
    if df.shape[1] > 1:
        int_columns = df.select_dtypes(include=['number']).columns[:-1]  # 去掉最后一列
        int_columns = [col for col in int_columns if col != exclude_column]  # 进一步排除指定列
    else:
        int_columns = df.select_dtypes(include=['number']).columns  # 如果只有一列，则全部处理
    
    # 仅处理整数列（不包括最后一列和指定列）
    for col in int_columns:
        df[col] = df[col].apply(round_with_1_percent_error)
    
    # 恢复数据类型
    for col in df.columns:
        df[col] = df[col].astype(original_dtypes[col])
    
    return df
    # # 保存回 CSV
    # df.to_csv(output_file, index=False)
    # print(f"处理完成，数据已保存至 {output_file}")

