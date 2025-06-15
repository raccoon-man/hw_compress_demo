import pandas as pd
import fastparquet
import os
import csv
import struct
import math
import numpy as np
from collections import Counter
import glob
import pyarrow as pa
import pyarrow.parquet as pq
import json
import segmentpro
import string
import lzma
import tarfile
import re
import zgk_pro
import trietree
from typing import Any
import time
from contextlib import contextmanager

# 性能分析相关变量
performance_stats = {
    "file_read": [],
    "pretreatment": {
        "total": 0,
        "per_column": {},
        "type_check": 0,
        "cardinality": 0,
        "segment": 0,
        "extract": 0,
        "concat": 0
    },
    "dict_process": {
        "total": 0,
        "per_column": {},
        "cardinality": 0,
        "compress": 0,
        "concat": 0
    },
    "compress_column": {
        "total": 0,
        "per_column": {},
        "filter_nulls": 0,     # 过滤空值
        "find_small_ints": 0,  # 查找小整数
        "count_frequency": 0,  # 统计频率
        "build_dict": 0,       # 构建字典
        "encode_data": 0       # 编码数据
    },
    "file_write": 0,
    "total": 0
}

@contextmanager
def timer(category, subcategory=None, column=None):
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    
    # 根据类别记录时间
    if subcategory is None and column is None:
        if isinstance(performance_stats[category], list):
            performance_stats[category].append(elapsed)
        else:
            performance_stats[category] = elapsed
    elif subcategory is not None and column is None:
        performance_stats[category][subcategory] += elapsed
    elif subcategory is not None and column is not None:
        if column not in performance_stats[category][subcategory]:
            performance_stats[category][subcategory][column] = 0
        performance_stats[category][subcategory][column] += elapsed

def median_cardinality(df: pd.DataFrame) -> int:
    """
    计算 DataFrame 中所有列的基数（unique 值数量），
    只统计基数 >= 10 的列，并返回这些列基数的中位数（整数）。
    
    若无满足条件的列，返回 0。
    """
    with timer("pretreatment", "cardinality"):
        cardinalities = df.nunique(dropna=True)
        filtered = cardinalities[cardinalities >= 10]
        
        if filtered.empty:
            return 0  # 没有满足条件的列时返回 0
        
        return int(filtered.median())

def compress_folder_to_xz(folder_path, output_xz_file):
    with timer("file_write"):
        # 使用 tarfile 将文件夹压缩为 tar 文件
        with tarfile.open(output_xz_file.replace('.xz', '.tar'), 'w') as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))
        
        # 使用 lzma 将 tar 文件压缩为 xz 格式
        with open(output_xz_file.replace('.xz', '.tar'), 'rb') as f_in:
            with lzma.open(output_xz_file, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # 删除临时的 tar 文件
        os.remove(output_xz_file.replace('.xz', '.tar'))

def get_cardinality(column_data):
    """
    计算列数据的基数（唯一值的数量），包括空值、None、'null' 和 NaN 的处理。
    """
    with timer("pretreatment", "cardinality"):
        # 如果 column_data 是列表，将其转换为 pandas Series 以便处理 NaN
        if isinstance(column_data, list):
            column_data = pd.Series(column_data)
        
        # 去除空值、None、'null' 和 NaN 的影响
        unique_values = set(column_data.replace(['', 'null'], None).drop_duplicates())

        # 如果唯一值集合为空，说明全是空值、None、'null' 或 NaN，将基数设置为 1
        if len(unique_values) == 0:
            return 1
        
        return len(unique_values)

def pretreatment(data, file_name):
    global json_size
    global df_preprocessed

    with timer("pretreatment", "total"):
        # 在处理前预先收集所有列信息，以便预分配空间
        with timer("pretreatment", "type_check"):
            dtype_dict = data.dtypes.to_dict()
            dtype_dict = {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()}

            # 收集需要处理的列
            columns_to_process = set(data.columns)

            # 初始化一个字典来存储所有列的处理结果
            column_data_dict = {}

        pattern = r'^-?\d+;-?\d+(;-?\d+)*$'
        
        for column_name in data.columns:
            column_data = data[column_name]
            if get_cardinality(column_data) == 1:
                json_content = get_single_json_content(column_name, str(column_data[0]))
                add_to_json_list(total_json['single'], json_content)
                data = data.drop(column_name, axis=1)

        # 第一步：处理 pattern
        for column_name in data.columns:
            with timer("pretreatment", "per_column", column_name):
                column_data = data[column_name]

                with timer("pretreatment", "type_check"):
                    # 处理 pattern
                    matches_format = column_data.astype(str).str.match(pattern)
                if matches_format.any() and column_name != 'gridid':
                    with timer("pretreatment", "extract"):
                        # 调用 extract_process 获取处理后的列
                        extracted_columns = extract_process_to_dict(column_data, column_name)
                        # 合并到总字典中
                        column_data_dict.update(extracted_columns)

                else:
                    # 先将列存入字典，后续再处理分割
                    column_data_dict[column_name] = column_data

        # print(column_data_dict)
        # 初始化一个字典来存储所有列的处理结果
        seg_dict = {}
        # 第二步：进行分割处理
        for column_name in column_data_dict.keys():
            if column_name in column_data_dict:
                column_data = column_data_dict[column_name].astype(str).replace('<NA>', "").replace('None', "")
                
                if column_name != 'gridid':
                    max_length = column_data.str.len().max()
                    if get_cardinality(column_data) > 128 and max_length > 4:
                        column_data = column_data.fillna('').astype(str)
                        # print(column_data)

                        with timer("pretreatment", "segment"):
                            # 列切分
                            dfn = segmentpro.SegmentExecute(column_data, column_name)

                        for col_name in dfn.columns:
                            large_integer_column.append(col_name)

                        # 存储分割后的列到字典中
                        for col in dfn.columns:
                            seg_dict[col] = dfn[col]
                    else:
                        # 存储列到字典中
                        seg_dict[column_name] = column_data

        # 一次性创建 DataFrame，而不是多次拼接
        with timer("pretreatment", "concat"):
            df_preprocessed = pd.DataFrame(seg_dict)

    with timer("file_write"):
        df_preprocessed.to_csv(f'preprocessed_data/{file_name}-preprocessed.csv', index=False)
# def pretreatment(data, file_name):
#     global json_size
#     global df_preprocessed

#     with timer("pretreatment", "total"):
#         # 在处理前预先收集所有列信息，以便预分配空间
#         with timer("pretreatment", "type_check"):
#             dtype_dict = data.dtypes.to_dict()
#             dtype_dict = {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()}
            
#             # 收集需要处理的列
#             columns_to_process = set(data.columns)
            
#             # 初始化一个字典来存储所有列的处理结果
#             column_data_dict = {}
        
#         pattern = r'^-?\d+(;-?\d+)*$'

#         for column_name in data.columns:
#             with timer("pretreatment", "per_column", column_name):
#                 column_data = data[column_name]
                
#                 with timer("pretreatment", "cardinality"):
#                     cardinal = get_cardinality(column_data)
                
#                 with timer("pretreatment", "type_check"):
#                     is_integer = pd.api.types.is_integer_dtype(column_data)
                
#                 if is_integer and column_name != 'gridid':
#                     min_value = column_data.min()
#                     # 判断基数是否大于数据总数的50%且数值长度普遍大于6
#                     if cardinal >= 128 and pd.notna(min_value) and min_value > 0:
#                         column_data = column_data.astype(str).replace('<NA>', "")
#                     # max_length = column_data.str.len().max()
#                     # if max_length > 4:
                        
#                         with timer("pretreatment", "segment"):
#                             # 列切分
#                             dfn = segmentpro.SegmentExecute(column_data, column_name)
                        
#                         for col_name in dfn.columns:
#                             large_integer_column.append(col_name)
                            
#                         # 存储分割后的列到字典中
#                         for col in dfn.columns:
#                             column_data_dict[col] = dfn[col]
#                     else:
#                         # 存储列到字典中
#                         column_data_dict[column_name] = column_data
#                 else:
#                     with timer("pretreatment", "type_check"):
#                         # 处理pattern
#                         matches_format = column_data.astype(str).str.match(pattern)
                    
#                     if matches_format.any() and column_name != 'gridid':
#                         with timer("pretreatment", "extract"):
#                             # 调用extract_process获取处理后的列
#                             extracted_columns = extract_process_to_dict(column_data, column_name)
#                             # 合并到总字典中
#                             column_data_dict.update(extracted_columns)
#                     else:
#                         # 存储列到字典中
#                         column_data_dict[column_name] = column_data
        
#         # 一次性创建DataFrame，而不是多次拼接
#         with timer("pretreatment", "concat"):
#             df_preprocessed = pd.DataFrame(column_data_dict)
    
#     with timer("file_write"):
#         df_preprocessed.to_csv(f'preprocessed_data/{file_name}-preprocessed.csv', index=False)

# 修改extract_process为返回字典格式，而不是直接拼接到df_preprocessed
def extract_process_to_dict(column_data, column):
    """处理含有分号的列，返回处理后的列字典"""
    result_dict = {}
    
    with timer("pretreatment", "extract"):
        # 处理Int64类型的空值问题
        if pd.api.types.is_integer_dtype(column_data):
            # 对整数列使用None而不是空字符串
            column_data = column_data.fillna(np.nan)
        else:
            column_data = column_data.fillna('') 
        
        # 确保 column_data 是字符串类型
        column_data = column_data.astype(str)
        
        # 按分号分割数据
        df_split = column_data.str.split(';', expand=True)
        df_split = df_split.applymap(lambda x: x if x and x not in '' else None)
        
        # 创建数值列
        df_values = df_split.applymap(lambda x: x if x else None)

        # 添加以下代码：尝试将所有数值统一转换为整数（如果可能）
        def convert_to_int_if_possible(x):
            if x is None:
                return None
            try:
                float_val = float(x)
                # 如果是整数值（如9.0），转换为整数
                if float_val.is_integer():
                    return str(int(float_val))
                return x
            except (ValueError, TypeError):
                return x

        df_values = df_values.applymap(convert_to_int_if_possible)

        # 创建分号列
        df_separators = df_split.applymap(lambda x: ';' if x not in [None, float('nan')] and pd.notna(x) else None)

        # 遍历分割后的列并添加到结果字典
        for i in range(df_values.shape[1]):
            # 处理数值列
            new_values = []
            for value in df_values.iloc[:, i]:
                try:
                    new_values.append(value)
                except (ValueError, TypeError):
                    new_values.append(None)
            result_dict[f'{column}_value_{i+1}'] = pd.Series(new_values)
            
            # 处理分号列，注意分号列比数值列少一列，且跳过最后一个分号
            if i < df_separators.shape[1] - 1:
                new_separators = df_separators.iloc[:, i]
                result_dict[f'{column}_separator_{i+1}'] = new_separators
    return result_dict

# 保留原有的extract_process函数以兼容其他代码
def extract_process(column_data, column):
    global df_preprocessed
    
    # 获取处理后的列字典
    result_dict = extract_process_to_dict(column_data, column)
    
    # 将列添加到df_preprocessed
    with timer("pretreatment", "concat"):
        if not df_preprocessed.empty:
            # 创建临时DataFrame
            temp_df = pd.DataFrame(result_dict)
            df_preprocessed = pd.concat([df_preprocessed, temp_df], axis=1)
        else:
            df_preprocessed = pd.DataFrame(result_dict)

BASE_CHARS = [chr(i) for i in range(32, 127) if chr(i) not in {',', ' ', 'N', 'A', ';', ':'}]  # 共 95 个字符
def int_to_ascii_key(n, base_chars=BASE_CHARS):
    """整数 → ASCII 编码字符串（变长编码）"""
    if n < 0:
        raise ValueError("Only non-negative integers are allowed")
    base = len(base_chars)
    if n == 0:
        return base_chars[0]
    encoded = ''
    while n > 0:
        n, rem = divmod(n, base)
        encoded = base_chars[rem - 1] + encoded
    return encoded

def compress_column(column_data, output_csv_file, column_name, top_n):
    with timer("compress_column", "total"):
        with timer("compress_column", "per_column", column_name):
            def is_effective_value(v):
                if pd.isna(v):
                    return False
                if isinstance(v, str) and v.strip().lower() == 'null':
                    return False
                if v == '':
                    return False
                return True
            
            # 过滤非空值（保留空占位）
            with timer("compress_column", "filter_nulls"):
                if not isinstance(column_data, pd.Series):
                    column_series = pd.Series(column_data)
                else:
                    column_series = column_data

                # 创建掩码: 非NA且不在指定值列表中
                mask = ~column_series.isna() & ~column_series.isin(['', 'null'])
                # 应用掩码获取结果
                non_null_data = column_series[mask].tolist()

            # 找出原始数据中出现的一位或两位整数值（以字符串形式存储，用于排除）
            with timer("compress_column", "find_small_ints"):
                small_int_strs = set()
                for v in non_null_data:
                    try:
                        if isinstance(v, int) and 0 <= v <= 99:
                            small_int_strs.add(str(v))
                        elif isinstance(v, str):
                            stripped = v.strip()
                            if stripped.isdigit() and 0 <= int(stripped) <= 99:
                                small_int_strs.add(stripped)
                    except:
                        continue

            # 统计 top_n 频率最高的值
            with timer("compress_column", "count_frequency"):
                frequency = Counter(non_null_data)
                most_common = frequency.most_common(top_n)
                most_common_values = [val for val, _ in most_common]

            # 构建编码字典：原值 -> ASCII 编码字符串
            with timer("compress_column", "build_dict"):
                value_dict = {}
                i = 0
                for val in most_common_values:
                    while True:
                        encoded = int_to_ascii_key(i)
                        if encoded not in small_int_strs:
                            value_dict[val] = encoded
                            break
                        i += 1
                    i += 1

                # top_n 占比
                top_n_count = sum(count for _, count in most_common)
                total_count = len(non_null_data) if non_null_data else 1
                top_n_percentage = top_n_count / total_count

            # 编码列数据 - 优化版本
            with timer("compress_column", "encode_data"):
                # 检查是否是pandas Series
                if isinstance(column_data, pd.Series):
                    # 使用pandas的向量化操作
                    # 1. 创建一个函数处理单个值
                    def encode_value(v):
                        if not is_effective_value(v):
                            return None
                        return value_dict.get(v, v)  # 不在字典中的值保持原样
                    
                    # 2. 向量化应用到整个Series
                    compressed_column = column_data.map(encode_value)
                else:
                    # 如果不是Series，先转换为Series再处理
                    series_data = pd.Series(column_data)
                    
                    # 使用Series.map进行向量化操作
                    compressed_column = series_data.map(lambda v: None if not is_effective_value(v) else value_dict.get(v, v))
                
                # 创建DataFrame
                df = pd.DataFrame({column_name: compressed_column})
            
            return df, value_dict, top_n_percentage

def get_json_content(column_name, dict):
    json_content = {}
    json_content[column_name] = dict
    return json_content

def get_single_json_content(column_name, value):
    json_content = {}
    json_content[column_name] = value
    return json_content

def contains_integer_range(column_data, lower_bound=0, upper_bound=15):
    return column_data.apply(lambda x: isinstance(x, int) and lower_bound <= x <= upper_bound).any()

# 计算最大位宽的函数
def get_max_width(column_data):
    # 确保传入的是 Series 对象
    if not isinstance(column_data, pd.Series):
        raise ValueError("输入必须是 pandas.Series 对象。")
    # 将 Series 中的元素转换为字符串，去除负号后计算最大长度
    max_width = column_data.astype(str).str.replace('-', '', regex=False).str.len().max()
    return max_width


def dict_process(colmn, file_name):
    global df_concatenated 
    global json_size
    global exceptions
    global cardinal_content
    global data_median_cardinality
    
    with timer("dict_process", "per_column", colmn):
        column_data = df_preprocessed[colmn]
        with timer("dict_process", "cardinality"):
            cardinal = get_cardinality(column_data)

        max_width = get_max_width(column_data)
        column_df = pd.DataFrame(column_data)
        contains_range = False
        
        # 如果基数为1，直接处理
        if cardinal == 1: 
            json_content = get_single_json_content(colmn, str(column_data[0]))
            add_to_json_list(total_json['single'], json_content)
            return 
        
        if max_width > 1 and cardinal < 16384:
            # 字典压缩
            with timer("dict_process", "compress"):
                df, dict, top_8_percentage = compress_column(column_data, f'city0/{file_name}-%s-compress.csv'%colmn, colmn, cardinal)
            
            json_content = get_json_content(colmn, dict)
            add_to_json_list(total_json['dict'], json_content)

            # 存储当前列的数据，等待后续一次性构建DataFrame
            compressed_columns[colmn] = df[colmn]
        else:
            compressed_columns[colmn] = column_data
    

def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


prefixes = ['city_', 'cell_', 'site_', 'country_']
def remove_prefix(value):
    if isinstance(value, str):
        for prefix in prefixes:
            if value.startswith(prefix):
                return value[len(prefix):]
    return value

def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(key): convert_keys_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    else:
        return obj

def add_to_dict(d: dict, key, value):
    """
    向字典 d 中添加一个键值对，其中 value 可以是任何类型。
    """
    d[key] = value
    return d

def add_to_json_list(json_list: list, element: Any):
    """
    向 JSON 列表（本质是 Python list）中添加一个元素（通常是字典）。
    """
    json_list.append(element)
    return json_list

def convert_to_nullable_int(df):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            # 判断该列是否全是整数或 NaN（允许小数点为.0）
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype('Int64')
    return df

def shorten_column_name(col_name: str) -> str:
    """
    将原始列名缩写为短形式
    """
    parts = col_name.split("_")

    if col_name.startswith("head_"):
        if len(parts) == 4 and parts[2] in ("value", "separator"):
            kind = 'v' if parts[2] == "value" else 's'
            return f"h{parts[1]}{kind}{parts[3]}"
        elif len(parts) == 3:
            return f"h{parts[1]}_{parts[2]}"
        elif len(parts) == 2:
            return f"h{parts[1]}"

    elif col_name.startswith("body_"):
        if len(parts) == 4 and parts[2] in ("value", "separator"):
            kind = 'v' if parts[2] == "value" else 's'
            return f"b{parts[1]}{kind}{parts[3]}"
        elif len(parts) == 3:
            return f"b{parts[1]}_{parts[2]}"
        elif len(parts) == 2:
            return f"b{parts[1]}"

    return col_name

def save_custom_txt_from_json(total_json: dict, file_name: str):
    """
    将 total_json 中的 dict、single 和 no 部分格式化后写入 txt 文件。
    """
    with timer("file_write"):
        output_path = f'txt/{file_name}/{file_name}.txt'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            # 写入 dict 部分
            for col_dict in total_json.get('dict', []):
                col_name = list(col_dict.keys())[0]
                inner_dict = col_dict[col_name]
                prefix = shorten_column_name(col_name)
                mapping_str = ','.join(f'{k}:{v}' for k, v in inner_dict.items())
                file.write(f'{prefix},{mapping_str}\n')

            # 写入 single 部分
            for item in total_json.get('single', []):
                for key, value in item.items():
                    prefix = shorten_column_name(key)
                    file.write(f'{prefix}:{value}\n')

            # 写入 no 部分
            for col_name in total_json.get('no', []):
                prefix = shorten_column_name(col_name)
                file.write(f'{prefix}\n')

def save_df_metadata(df, filename):
    with timer("file_write"):
        # 构造输出路径
        output_path = os.path.join('metadata', f"{filename}.txt")

        # 如果文件已存在，跳过处理
        if os.path.exists(output_path):
            print(f"[跳过] {output_path} 已存在。")
            return

        # 替换规则
        dtype_map = {
            'int64': 'I',
            'Int64': 'I',
            'float64': 'F',
            'Float64': 'F',
            'object': 'O',
            'string': 'O'
        }

        # 构建 metadata 内容
        lines = []
        for col, dtype in df.dtypes.items():
            new_col = col.replace('head_', 'h').replace('body_', 'b')
            dtype_str = dtype_map.get(str(dtype), 'U')  # 'U' = Unknown
            lines.append(f"{new_col}:{dtype_str}")

        # 创建 metadata 目录
        os.makedirs('metadata', exist_ok=True)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"[完成] Metadata 写入至 {output_path}")

def generate_performance_report():
    """生成性能报告"""
    # 总体耗时
    total_time = performance_stats["total"]
    print(f"\n{'*'*30} 性能报告 {'*'*30}")
    print(f"总执行时间: {total_time:.2f} 秒")
    
    # 各阶段耗时
    print(f"\n{'='*20} 各阶段耗时 {'='*20}")
    print(f"文件读取: {sum(performance_stats['file_read']):.2f} 秒 ({sum(performance_stats['file_read'])/total_time*100:.1f}%)")
    print(f"预处理: {performance_stats['pretreatment']['total']:.2f} 秒 ({performance_stats['pretreatment']['total']/total_time*100:.1f}%)")
    
    # 字典处理耗时
    dict_time = sum(performance_stats['dict_process']['per_column'].values())
    print(f"字典压缩: {dict_time:.2f} 秒 ({dict_time/total_time*100:.1f}%)")
    
    # 文件写入耗时
    print(f"文件写入: {performance_stats['file_write']:.2f} 秒 ({performance_stats['file_write']/total_time*100:.1f}%)")
    
    # 预处理阶段细分
    print(f"\n{'='*20} 预处理阶段细分 {'='*20}")
    pretreatment_total = performance_stats['pretreatment']['total']
    print(f"类型检查: {performance_stats['pretreatment']['type_check']:.2f} 秒 ({performance_stats['pretreatment']['type_check']/pretreatment_total*100:.1f}%)")
    print(f"基数计算: {performance_stats['pretreatment']['cardinality']:.2f} 秒 ({performance_stats['pretreatment']['cardinality']/pretreatment_total*100:.1f}%)")
    print(f"大整数切分: {performance_stats['pretreatment']['segment']:.2f} 秒 ({performance_stats['pretreatment']['segment']/pretreatment_total*100:.1f}%)")
    print(f"分号处理: {performance_stats['pretreatment']['extract']:.2f} 秒 ({performance_stats['pretreatment']['extract']/pretreatment_total*100:.1f}%)")
    print(f"DataFrame拼接: {performance_stats['pretreatment']['concat']:.2f} 秒 ({performance_stats['pretreatment']['concat']/pretreatment_total*100:.1f}%)")
    
    # 字典压缩阶段细分
    print(f"\n{'='*20} 字典压缩阶段细分 {'='*20}")
    dict_process_total = dict_time + performance_stats['dict_process']['concat']
    print(f"基数计算: {performance_stats['dict_process']['cardinality']:.2f} 秒 ({performance_stats['dict_process']['cardinality']/dict_process_total*100:.1f}%)")
    print(f"压缩处理: {performance_stats['dict_process']['compress']:.2f} 秒 ({performance_stats['dict_process']['compress']/dict_process_total*100:.1f}%)")
    print(f"DataFrame拼接: {performance_stats['dict_process']['concat']:.2f} 秒 ({performance_stats['dict_process']['concat']/dict_process_total*100:.1f}%)")
    
    # 压缩处理细分
    print(f"\n{'='*20} 压缩处理细分 {'='*20}")
    compress_total = performance_stats['compress_column']['total']
    print(f"过滤空值: {performance_stats['compress_column']['filter_nulls']:.2f} 秒 ({performance_stats['compress_column']['filter_nulls']/compress_total*100:.1f}%)")
    print(f"查找小整数: {performance_stats['compress_column']['find_small_ints']:.2f} 秒 ({performance_stats['compress_column']['find_small_ints']/compress_total*100:.1f}%)")
    print(f"统计频率: {performance_stats['compress_column']['count_frequency']:.2f} 秒 ({performance_stats['compress_column']['count_frequency']/compress_total*100:.1f}%)")
    print(f"构建字典: {performance_stats['compress_column']['build_dict']:.2f} 秒 ({performance_stats['compress_column']['build_dict']/compress_total*100:.1f}%)")
    print(f"编码数据: {performance_stats['compress_column']['encode_data']:.2f} 秒 ({performance_stats['compress_column']['encode_data']/compress_total*100:.1f}%)")
    
    # 耗时最长的列 (预处理)
    print(f"\n{'='*20} 预处理耗时最长的列 (Top 5) {'='*20}")
    sorted_cols = sorted(performance_stats['pretreatment']['per_column'].items(), key=lambda x: x[1], reverse=True)
    for col, time_spent in sorted_cols[:5]:
        print(f"{col}: {time_spent:.2f} 秒 ({time_spent/pretreatment_total*100:.1f}%)")
    
    # 耗时最长的列 (字典压缩)
    print(f"\n{'='*20} 字典压缩耗时最长的列 (Top 5) {'='*20}")
    sorted_cols = sorted(performance_stats['dict_process']['per_column'].items(), key=lambda x: x[1], reverse=True)
    for col, time_spent in sorted_cols[:5]:
        print(f"{col}: {time_spent:.2f} 秒 ({time_spent/dict_process_total*100:.1f}%)")
    
    # 生成优化建议
    print(f"\n{'='*20} 优化建议 {'='*20}")
    bottlenecks = []
    
    if performance_stats['pretreatment']['concat'] / pretreatment_total > 0.3:
        bottlenecks.append("预处理阶段DataFrame拼接操作耗时较长，建议使用预分配DataFrame空间减少拼接次数")
        
    if performance_stats['pretreatment']['cardinality'] / pretreatment_total > 0.2:
        bottlenecks.append("基数计算耗时较长，建议使用缓存机制避免重复计算")
        
    if performance_stats['pretreatment']['extract'] / pretreatment_total > 0.25:
        bottlenecks.append("分号处理耗时较长，建议优化分隔符列的处理逻辑")
        
    if performance_stats['dict_process']['compress'] / dict_process_total > 0.6:
        bottlenecks.append("压缩处理耗时较长，建议优化字典构建和数据映射逻辑")
    
    if performance_stats['compress_column']['find_small_ints'] / compress_total > 0.3:
        bottlenecks.append("查找小整数操作耗时较长，考虑优化小整数查找算法或使用更高效的数据结构")
        
    if performance_stats['compress_column']['count_frequency'] / compress_total > 0.3:
        bottlenecks.append("频率统计耗时较长，考虑使用更高效的计数方法或预先过滤数据")
        
    if performance_stats['compress_column']['encode_data'] / compress_total > 0.4:
        bottlenecks.append("数据编码耗时较长，考虑优化编码查找逻辑或使用预计算的映射表")
    
    if performance_stats['dict_process']['concat'] / dict_process_total > 0.2:
        bottlenecks.append("字典压缩阶段DataFrame拼接耗时较长，建议检查优化策略是否生效")
    
    if dict_time / total_time > 0.5:
        bottlenecks.append("字典压缩阶段占用大量时间，可以考虑并行处理多列")
    
    for suggestion in bottlenecks:
        print(f"- {suggestion}")
    
    if not bottlenecks:
        print("- 没有明显的性能瓶颈，代码运行良好")

if __name__ == "__main__":
    # 设置全局参数，表示进行列压缩的次数

    with timer("total"):
        # 示例用法
        with timer("file_read"):
            csv_files = get_csv_files_without_extension('data/')
        
        print(csv_files)
        dimension = {}
        dimension['16843010'] = 'body_1005'
        dimension['16777219'] = 'head_12'
        dimension['16777220'] = 'head_12'
        
        for file_name in csv_files:
            with timer("file_read"):
                file_path = f'data/{file_name}.csv' 
                file_name = 'dict-' + file_name
                data = pd.read_csv(file_path).applymap(remove_prefix)
                data = convert_to_nullable_int(data)
                save_df_metadata(data, file_name)
            
            #大整数的列集合
            large_integer_column = []
            #统计字典的大小
            json_size = 0
            #未处理的列集合
            exceptions = []
            cardinal_content = {}
            total_json = {}
            total_json['dict'] = []
            total_json['single'] = []
            total_json['no'] = []
            
            part1, part2 = file_name.rsplit('-', 1)
            print(part1, part2)

            data_median_cardinality = median_cardinality(data)
            #最终df
            df_concatenated = pd.DataFrame()
            df_preprocessed = pd.DataFrame()
            df_test = pd.DataFrame()
            
            # 创建一个字典来存储压缩后的列，用于一次性构建DataFrame
            compressed_columns = {}

            # 创建json的文件夹
            output_dir = os.path.join(f'json/{file_name}')
            os.makedirs(output_dir, exist_ok=True)
            
            pretreatment(data, file_name)

            # 假设df_preprocessed是你的DataFrame
            columns_info = {col: str(df_preprocessed[col].dtype) for col in df_preprocessed.columns}
            
            with timer("dict_process", "total"):
                for column_name in df_preprocessed.columns:
                    dict_process(column_name, file_name)
            
            # 一次性构建压缩后的DataFrame
            with timer("dict_process", "concat"):
                df_concatenated = pd.DataFrame(compressed_columns)
                # df_concatenated = pd.DataFrame(df_preprocessed)
            
            #生成csv文件便于调试
            with timer("file_write"):
                df_concatenated.to_csv(f'compress_data/csv/{file_name}-compress.csv', index=False)
            
            total_json_str_keys = convert_keys_to_str(total_json)
            save_custom_txt_from_json(total_json_str_keys, file_name)
    
    # 生成性能报告
    # generate_performance_report() 