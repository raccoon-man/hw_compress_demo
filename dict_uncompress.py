import pandas as pd
import os
import json
import glob
import re
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings("ignore")
def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]

import re

def restore_column_name(short_name: str) -> str:

    # 处理 head_ 列（例如 h1, h15s3, h15v5, h15_0）
    if short_name.startswith("h"):
        # 匹配 head_ 列名，包括带下划线的情况
        match = re.match(r"h(\d+)([sv])(\d+)$", short_name)
        if match:
            num, flag, subnum = match.groups()
            suffix = "separator" if flag == "s" else "value"
            return f"head_{num}_{suffix}_{subnum}"
        
        # 处理带下划线的列名（例如 h13_0）
        match = re.match(r"h(\d+)_(\d+)$", short_name)
        if match:
            num, subnum = match.groups()
            return f"head_{num}_{subnum}"

        # 处理类似 'h1' -> 'head_1'
        elif short_name[1:].isdigit():
            return "head_" + short_name[1:]

    # 处理 body_ 列（例如 b15, b15s3, b15v5, b15_0）
    elif short_name.startswith("b"):
        # 匹配 body_ 列名，包括带下划线的情况
        match = re.match(r"b(\d+)([sv])(\d+)$", short_name)
        if match:
            num, flag, subnum = match.groups()
            suffix = "separator" if flag == "s" else "value"
            return f"body_{num}_{suffix}_{subnum}"

        # 处理带下划线的列名（例如 b13_0）
        match = re.match(r"b(\d+)_(\d+)$", short_name)
        if match:
            num, subnum = match.groups()
            return f"body_{num}_{subnum}"

        # 处理类似 'b15' -> 'body_15'
        elif short_name[1:].isdigit():
            return "body_" + short_name[1:]

    # 返回无法匹配的列名（不进行修改）
    return short_name



def load_custom_txt_to_json(file_name: str):
    """
    从 txt 文件读取并解析成 total_json 格式，并还原列名前缀。
    
    :param file_name: 文件名（不带路径）
    :return: total_json 字典
    """
    def restore_column_name(short_name):
        if short_name.startswith('h') and short_name[1:].isdigit():
            return f"head_{short_name[1:]}"
        elif short_name.startswith('b') and short_name[1:].isdigit():
            return f"body_{short_name[1:]}"
        else:
            return short_name

    input_path = f'txt/{file_name}/{file_name}.txt'
    total_json = {'dict': [], 'single': [], 'no': []}

    with open(input_path, 'r', encoding='utf-8') as file:
        current_section = None
        for line in file:
            line = line.strip()
            if not line:
                continue

            if ',' in line:
                current_section = 'dict'
            elif ':' in line:
                current_section = 'single'
            else:
                current_section = 'no'

            if current_section == 'dict':
                col_name, mappings = line.split(',', 1)
                col_name = restore_column_name(col_name)
                # 注意：拆分时要确保每一项仅包含一个冒号
                item_pairs = [item.split(':', 1) for item in mappings.split(',') if ':' in item]
                mappings = dict(item_pairs)
                total_json['dict'].append({col_name: mappings})

            elif current_section == 'single':
                key, value = line.split(':', 1)
                key = restore_column_name(key)
                total_json['single'].append({key: value})

            elif current_section == 'no':
                line = restore_column_name(line)
                total_json['no'].append(line)

    return total_json


def insert_column_in_order(df, column_name, value):
    """
    如果 df 中没有 column_name，则创建一个值全为 value 的新列，
    并按 head_ 和 body_ 分组，编号升序插入，避免内存碎片化。
    """

    def get_order_key(col):
        match = re.match(r"(head|body)_(\d+)", col)
        if match:
            prefix, number = match.groups()
            return (0 if prefix == 'head' else 1, int(number))
        return (2, float('inf'))

    # 如果列已存在，直接返回原 DataFrame
    if column_name in df.columns:
        return df

    # 构造新列（Series），索引与原 df 一致
    new_col_df = pd.DataFrame({column_name: [value] * len(df)}, index=df.index)

    # 临时合并列（不会立即改变顺序）
    df_combined = pd.concat([df, new_col_df], axis=1)

    # 重新排序列
    all_columns = df_combined.columns.tolist()
    all_columns_sorted = sorted(all_columns, key=get_order_key)

    # 返回排序后的新 DataFrame，避免原始 df 过多 insert 造成碎片
    return df_combined[all_columns_sorted]

def decompress_column(df, total_json, column_name):
    """
    解压缩 CSV 数据，按需恢复 'dict' 和 'single' 部分。

    :param df: 输入的 DataFrame（CSV 数据）。
    :param total_json: 从 .txt 文件恢复的 total_json 字典。
    :param column_name: 当前列的列名。
    :return: 解压后的 DataFrame。
    """
    # 获取 dict 部分映射

    dict_mappings = None
    for col_dict in total_json.get('dict', []):
        for key in col_dict:
            restored_name = restore_column_name(key)
            if restored_name == column_name:
                dict_mappings = col_dict[key]
                break
        if dict_mappings:
            break
    
    # dict 类型列还原
    if dict_mappings:
        reverse_dict = {v: k for k, v in dict_mappings.items()}
        df[column_name] = df[column_name].apply(lambda x: reverse_dict.get(x, x))
    return df

# 主函数：加载 txt 和解压缩 CSV
def decompress_csv_with_txt(csv_file, txt_file):
    """
    根据 txt 文件中的压缩字典解压缩给定的 CSV 文件。
    
    :param csv_file: 输入的 CSV 文件路径。
    :param txt_file: 输入的 txt 文件（包含压缩字典）。
    :return: 解压缩后的 DataFrame。
    """
    # 加载压缩字典信息
    total_json = load_custom_txt_to_json(txt_file)
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file, dtype=str)

    df = df.where(pd.notna(df), "")
    # 遍历 DataFrame 的列，解压每列
    for column in df.columns:
        df = decompress_column(df, total_json, column)

        # single 类型列插入
    for item in total_json.get('single', []):
        for key, value in item.items():
            print(key, restore_column_name(key))
            df = insert_column_in_order(df, restore_column_name(key), value)

    return df

def auto_merge_complex_subcolumns(df):
    """
    自动识别并合并 body_x_value_y 和 body_x_separator_y 子列，
    合并顺序为：按子编号升序排列，每组内先拼接 value 后 separator，
    最后确保合并后的列也按照正确的顺序排列。
    """
    grouped_cols = defaultdict(list)

    # 识别所有符合格式的列，并分组
    for col in df.columns:
        m = re.match(r"^(body|head)_(\d+)_(value|separator)_(\d+)$", col)
        if m:
            prefix, group_id, kind, sub_id = m.groups()
            base_col = f"{prefix}_{group_id}"
            kind_priority = 0 if kind == "value" else 1  # value 优先
            grouped_cols[base_col].append((int(sub_id), kind_priority, col))

    # 合并各组列
    for base_col, entries in grouped_cols.items():
        # 排序规则：先按 sub_id，再按 kind_priority（value -> separator）
        entries.sort()
        sorted_cols = [col for _, _, col in entries]

        # 合并列内容
        df[base_col] = df[sorted_cols].astype(str).agg(''.join, axis=1)

        # 删除原始子列
        df.drop(columns=sorted_cols, inplace=True)
        df[base_col] = df[base_col].str.rstrip(';')

    # 排序 DataFrame 列，确保 head_x 在前，body_x 在后
    def get_order_key(col):
        m = re.match(r"^(head|body)_(\d+)$", col)
        if m:
            prefix, group_id = m.groups()
            return (0 if prefix == "head" else 1, int(group_id))
        return (2, float('inf'))  # 其他列排最后

    df = df[sorted(df.columns, key=get_order_key)]

    return df

def convert_columns_from_txt(txt_path, df):
    # 定义列名前缀映射和数据类型映射
    column_name_mapping = {'h': 'head', 'b': 'body'}
    dtype_mapping = {'I': 'Int64', 'O': 'object'}  # pandas 支持缺失值的整数类型

    # 读取并解析 txt 文件
    with open(txt_path, 'r') as f:
        lines = f.read().splitlines()

    rename_and_types = {}
    ordered_columns = []  # 用于保存顺序
    
    for line in lines:
        if ":" not in line:
            continue
        short_name, dtype_code = line.split(":")
        prefix = short_name[0]
        index = short_name[1:]
        full_name = f"{column_name_mapping.get(prefix, prefix)}_{index}"
        pandas_dtype = dtype_mapping.get(dtype_code, dtype_code)
        rename_and_types[full_name] = pandas_dtype
        ordered_columns.append(full_name)

    # 转换对应列的数据类型
    for col, dtype in rename_and_types.items():
        if col in df.columns:
            # 将字符串 "<NA>" 和空字符串替换为真正的 pd.NA
            # print(df[col])
            df[col] = df[col].replace(["<NA>", "", "NaN","None"], pd.NA)
            try:
                # print(df[col])
                df[col] = df[col].astype(dtype)
            except Exception as e:
                # print(df[col])

                print(f"⚠️ 列 {col} 转换为 {dtype} 时出错：{e}")

    # 重新排列列顺序（只排列在 txt 中定义过的列，其它列保持在后面）
    ordered_cols_in_df = [col for col in ordered_columns if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols_in_df]
    df = df[ordered_cols_in_df + remaining_cols]

    return df



def auto_merge_prefix_subcolumns(df):
    """
    自动识别并合并所有匹配 head_x_y 或 body_x_y 格式的列，
    合并后列名为 head_x 或 body_x。

    :param df: 原始 DataFrame。
    :return: 合并后的 DataFrame。
    """
    grouped_cols = defaultdict(list)
    pattern = re.compile(r"^((head|body)_\d+)_(\d+)$")  # 支持 head_x_y 和 body_x_y

    for col in df.columns:
        match = pattern.match(col)
        if match:
            base_col, _, sub = match.groups()
            grouped_cols[base_col].append((int(sub), col))

    for base_col, subcols in grouped_cols.items():
        subcols.sort()
        sorted_cols = [col for _, col in subcols]
        df[base_col] = df[sorted_cols].astype(str).agg(''.join, axis=1)
        # print(df[base_col])
        df.drop(columns=sorted_cols, inplace=True)

        # 排序所有列，确保合并后的列按预期顺序排列
    def get_order_key(col):
        # 支持 head_14、head_13_1、body_15_value_5 等复杂字段排序
        match = re.match(r"(head|body)_(\d+)", col)
        if match:
            prefix, number = match.groups()
            return (0 if prefix == 'head' else 1, int(number))
        return (2, float('inf'))  # 非 head 或 body 列排在最后

    # 获取所有列并排序
    cols = df.columns.tolist()
    cols_sorted = sorted(cols, key=get_order_key)

    # 重新排序 DataFrame 列
    df = df[cols_sorted]

    return df


def convert_to_nullable_int(df):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            # 判断该列是否全是整数或 NaN（允许小数点为.0）
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype('Int64')
    return df


def compare_csv_content(file1, file2, tol=1e-9):
    """
    严格比较两个CSV文件的内容是否相同（列名顺序、形状、值都一致）。
    特殊处理了 '<NA>'、''、'NaN' 等常见缺失值，支持 nullable int 比较。
    """
    try:
        # 读取 CSV 文件
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df1 = convert_to_nullable_int(df1)
        df2 = convert_to_nullable_int(df2)



        # 比较列名顺序
        if list(df1.columns) != list(df2.columns):
            print("❌ 列名不一致")
            print("file1 columns:", list(df1.columns))
            print("file2 columns:", list(df2.columns))
            return False

        # 比较形状
        if df1.shape != df2.shape:
            print(f"❌ 行列数不同：file1 是 {df1.shape}，file2 是 {df2.shape}")
            return False



        # 比较内容：逐元素比较
        diff_found = False  # 用于标记是否发现差异
        for row in range(df1.shape[0]):  # 遍历行
            for col in df1.columns:  # 遍历列
                val1 = df1.at[row, col]
                val2 = df2.at[row, col]

                # 特别处理缺失值的比较
                if pd.isna(val1) and pd.isna(val2):  # 如果两个值都是 NA 或 NaN，认为相等
                    continue
                if pd.isna(val1) or pd.isna(val2):  # 如果只有一个是 NA 或 NaN，认为不相等
                    # print(f"❌ 内容不同：行 {row}，列 '{col}'，file1 = {val1!r}, file2 = {val2!r}")
                    diff_found = True
                    continue

                # 对其他类型的值进行普通的比较
                if val1 != val2:  # 如果值不同
                    # print(f"❌ 内容不同：行 {row}，列 '{col}'，file1 = {val1!r}, file2 = {val2!r}")
                    diff_found = True

        if not diff_found:  # 如果没有发现差异
            print("✅ 字典解压缩文件和原文件一致")
            return True
        return False


    except Exception as e:
        print(f"⚠️ 比较出错：{e}")
        return False




if __name__ == "__main__":
    csv_files = get_csv_files_without_extension('compress_data/csv/')
        # 示例：解压缩 CSV
    print(csv_files)
    for file_name in csv_files:
        print(file_name)
        parts = file_name.split('-')
        original_file_name = '-'.join(parts[:2][1:]) 
        original_file_path = f'data/{original_file_name}.csv'
        # print(original_file_path)
        txt_file_name = '-'.join(parts[:2]) 
        metadata_file_name = f'metadata/{txt_file_name}.txt'
        # print(txt_file_name)
        file_path = f'compress_data/csv/{file_name}.csv' 
        folder_path = 'uncompress_dict/'
        os.makedirs(folder_path, exist_ok=True)
        uncompress_file_path = os.path.join(folder_path, f'{original_file_name}-dict-uncompress.csv')


        df_decompressed = decompress_csv_with_txt(file_path, txt_file_name)

        df_decompressed = auto_merge_prefix_subcolumns(df_decompressed)

        df_decompressed = auto_merge_complex_subcolumns(df_decompressed)

        df_decompressed = convert_columns_from_txt(metadata_file_name, df_decompressed)

        df_decompressed.to_csv(uncompress_file_path,index = False)
        compare_csv_content(uncompress_file_path, original_file_path)
