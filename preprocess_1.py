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
import dict as dickWork 

def process_special_values(data, file_name, total_json):
    """
    处理特殊大整数和空值替换
    Args:
        data: DataFrame，原始数据
        file_name: str，文件名
        total_json: dict，用于存储处理信息的字典
    Returns:
        DataFrame: 处理后的数据
    """
    # 记录特殊大整数替换的列
    large_int_dict = {}
    # 记录空值替换的信息
    null_value_dict = {}

    # 遍历所有列
    for column_name in data.columns:
        column_data = data[column_name]
        
        # 1. 特殊大整数清洗
        if pd.api.types.is_numeric_dtype(column_data):
            # 转换为字符串进行比较，避免精度问题
            str_data = column_data.astype(str)
            if (str_data == '65535').any() or (str_data == '4294967295').any() or (str_data == '2147483647').any():
                # 记录这个列进行了大整数替换
                replacements = {}
                if (str_data == '65535').any():
                    replacements['65535'] = '-1'
                if (str_data == '4294967295').any():
                    replacements['4294967295'] = '-1'
                if (str_data == '2147483647').any():
                    replacements['2147483647'] = '1'
                large_int_dict[column_name] = replacements
                
                # 执行替换
                column_data = column_data.replace({
                    65535: -1,
                    4294967295: -1,
                    2147483647: 1
                })
                data[column_name] = column_data

        # 2. 空值替换
        if column_data.isna().any():
            # 检查是否存在 -2
            has_minus_two = False
            if pd.api.types.is_numeric_dtype(column_data):
                has_minus_two = (column_data == -2).any()
            
            if has_minus_two:
                # 找到最小值并减1
                min_value = column_data.dropna().min()
                replacement_value = min_value - 1
            else:
                replacement_value = -2
            
            # 记录空值替换信息
            null_value_dict[column_name] = str(replacement_value)
            # 执行替换
            data[column_name] = column_data.fillna(replacement_value)

    # 将特殊大整数替换信息添加到 total_json
    if large_int_dict:
        total_json['large_int_replacements'] = large_int_dict

    # 将空值替换信息添加到 total_json
    if null_value_dict:
        total_json['null_values'] = null_value_dict

    return data

def pretreatment(data, file_name):
    global json_size
    global df_preprocessed
        # 在处理前预先收集所有列信息，以便预分配空间
    dtype_dict = data.dtypes.to_dict()
    dtype_dict = {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()}

    # 收集需要处理的列
    columns_to_process = set(data.columns)

        # 初始化一个字典来存储所有列的处理结果
    column_data_dict = {}

    pattern = r'^-?\d+;-?\d+(;-?\d+)*$'
    
    for column_name in data.columns:
        column_data = data[column_name]
        if dickWork.get_cardinality(column_data) == 1:
            json_content = dickWork.get_single_json_content(column_name, str(column_data[0]))
            dickWork.add_to_json_list(total_json['single'], json_content)
            data = data.drop(column_name, axis=1)

    # 第一步：处理 pattern
    for column_name in data.columns:
            column_data = data[column_name]

                # 处理 pattern
            matches_format = column_data.astype(str).str.match(pattern)
            if matches_format.any() and column_name != 'gridid':
                    # 调用 extract_process 获取处理后的列
                    extracted_columns = dict.extract_process_to_dict(column_data, column_name)
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
                if dickWork.get_cardinality(column_data) > 128 and max_length > 4:
                    column_data = column_data.fillna('').astype(str)
                    # print(column_data)
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


    df_preprocessed = pd.DataFrame(seg_dict)

    df_preprocessed.to_csv(f'preprocessed_data/{file_name}-preprocessed.csv', index=False)

if __name__ == "__main__":
    # 设置全局参数，表示进行列压缩的次数
    csv_files = dickWork.get_csv_files_without_extension('data/')
    
    print(csv_files)
    dimension = {}
    dimension['16843010'] = 'body_1005'
    dimension['16777219'] = 'head_12'
    dimension['16777220'] = 'head_12'
    
    for file_name in csv_files:
        file_path = f'data/{file_name}.csv' 
        file_name = 'dict-' + file_name
        data = pd.read_csv(file_path).applymap(dickWork.remove_prefix)
        data = dickWork.convert_to_nullable_int(data)
        dickWork.save_df_metadata(data, file_name)
        
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
        total_json['null_values'] = {}  # 空值替换信息的字典
        total_json['large_int_replacements'] = {}  # 特殊大整数替换信息的字典
        
        part1, part2 = file_name.rsplit('-', 1)
        print(part1, part2)

        # 处理特殊大整数和空值
        data = process_special_values(data, file_name, total_json)

        data_median_cardinality = dickWork.median_cardinality(data)
        #最终df
        df_concatenated = pd.DataFrame()
        df_preprocessed = pd.DataFrame()
        
        # 创建一个字典来存储压缩后的列，用于一次性构建DataFrame
        compressed_columns = {}

        # 创建json的文件夹
        output_dir = os.path.join(f'json/{file_name}')
        os.makedirs(output_dir, exist_ok=True)
        
        pretreatment(data, file_name)

        # 假设df_preprocessed是你的DataFrame
        columns_info = {col: str(df_preprocessed[col].dtype) for col in df_preprocessed.columns}
        
        #生成csv文件便于调试
        df_preprocessed.to_csv(f'compress_data/csv/{file_name}-compress.csv', index=False)
        
        total_json_str_keys = dickWork.convert_keys_to_str(total_json)
        dickWork.save_custom_txt_from_json(total_json_str_keys, file_name)