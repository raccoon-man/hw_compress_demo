import pandas as pd
import fastparquet
import os
import matplotlib.pyplot as plt
import csv
import struct
import math
import numpy as np
from collections import Counter
import glob
import pyarrow as pa
import pyarrow.parquet as pq
import json
import segment
import string






def get_cardinality(column_data):
    """
    计算列数据的基数（唯一值的数量），包括空值、None、'null' 和 NaN 的处理。

    参数:
        column_data (list or pd.Series): 列数据列表或 Series。

    返回:
        int: 基数，即唯一值的数量。如果只有空值、None、'null' 或 NaN，基数为 1。
    """
    # 如果 column_data 是列表，将其转换为 pandas Series 以便处理 NaN
    if isinstance(column_data, list):
        column_data = pd.Series(column_data)
    
    # 去除空值、None、'null' 和 NaN 的影响
    unique_values = set(column_data.replace(['', 'null'], None).drop_duplicates())

    # 如果唯一值集合为空，说明全是空值、None、'null' 或 NaN，将基数设置为 1
    if len(unique_values) == 0:
        return 1
    
    return len(unique_values)


#预处理 将长整型拆分，带分号的拆分
def pretreatment(data, file_name):
    global json_size
    global df_preprocessed

    dtype_dict = data.dtypes.to_dict()
    dtype_dict = {col: str(dtype) for col, dtype in data.dtypes.to_dict().items()}
    dict_filename = f'json/{file_name}/{file_name}-dtypes.json'
    with open(dict_filename, 'w') as json_file:
        json.dump(dtype_dict, json_file)
    json_size = json_size + os.path.getsize(dict_filename)


    pattern = r'^-?\d+(;-?\d+)*$'

    for column_name in data.columns:
        column_data = data[column_name]
        cardinal = get_cardinality(column_data)
        if pd.api.types.is_integer_dtype(column_data):
            min_value = column_data.min()
            # 判断基数是否大于数据总数的50%且数值长度普遍大于6
            if cardinal > 0.3 * len(column_data) and min_value > 0:
                column_data = column_data.astype(str)
                # print(file_name,column_name,cardinal)
                dfn = segment.SegmentExecute(column_data, column_name)
                for column_name in dfn.columns:
                    large_integer_column.append(column_name)
                df_preprocessed = pd.concat([df_preprocessed, dfn], axis=1)
            else : 
                df_preprocessed = pd.concat([df_preprocessed, column_data], axis=1)
        else:
            matches_format = column_data.astype(str).str.match(pattern)
            if matches_format.any():
                extract_process(column_data, column_name)
            else : 
                df_preprocessed = pd.concat([df_preprocessed, column_data], axis=1)
    
    # print(large_integer_column)
    df_preprocessed.to_csv(f'preprocessed_data/{file_name}-preprocessed.csv', index=False)
    

#字典压缩
def compress_column(column_data, output_csv_file, column_name):
    """
    对给定的列数据进行字典压缩，并将压缩后的结果保存到CSV文件中。
    
    参数:
        column_data (list): 要压缩的列数据列表。
        output_csv_file (str): 压缩后保存的CSV文件路径。
        column_name (str): 在输出CSV文件中的列名，默认为 'compressed_column'。
    
    返回:
        dict: 压缩字典，用于查看原始值与编码值的映射关系。
    """
    if all(isinstance(value, int) or value in ('', None, 'null') for value in column_data):
        value_type = int
    else:
        value_type = str

    # Step 2: 统计非null值的频率
    non_null_data = [value for value in column_data if value not in ('', None, 'null')]
    frequency = Counter(non_null_data)

    # Step 3: 找出出现频率最高的前十六个元素
    most_common = frequency.most_common(16)
    value_dict = {value: idx for idx, (value, _) in enumerate(most_common)}
    
    # 计算前8个元素占总元素的比例
    top_8_count = sum(count for _, count in most_common)
    total_count = len(non_null_data) if non_null_data else 1  # 避免除零错误
    top_8_percentage = top_8_count / total_count

    # 为null值分配特定的编码值
    null_encoding = -1 if value_type is int else '-1'

    # Step 4: 压缩列数据并确保数据类型一致
    compressed_column = []
    for value in column_data:
        if value in ('', None, 'null'):
            compressed_column.append(null_encoding)  # 编码null值
        elif value in value_dict:
            if value_type is int:
                compressed_column.append(value_dict[value])  # 使用整数字典编码
            else:
                compressed_column.append(str(value_dict[value]))  # 使用字符串字典编码
        else:
            compressed_column.append(value_type(value))

    df = pd.DataFrame({column_name: compressed_column})


    return df, value_dict, top_8_percentage


def get_json_content(compress_type, dict, value):
    json_content = {}
    json_content["compress_type"] = compress_type
    json_content["dict"] = dict
    json_content["value"] = value
    return json_content

def process_independence_column(colmn, file_name):
        global exceptions
        global json_size
        exceptions.append(colmn)
        json_content = get_json_content('no_process', None, None)
                 # 获取 JSON 文件的大小（统计字典json 文件大小）
        dict_filename = f'json/{file_name}/{file_name}-%s-compress-dict.json' % colmn
        with open(dict_filename, 'w') as json_file:
            json.dump(json_content, json_file)
        json_size = json_size + os.path.getsize(dict_filename)

def contains_integer_range(column_data, lower_bound=0, upper_bound=15):
    # 使用.apply()方法结合lambda表达式检查每个元素是否在指定范围内
    return column_data.apply(lambda x: isinstance(x, int) and lower_bound <= x <= upper_bound).any()

#分类处理每一列
def dict_process(colmn, file_name):
    global df_concatenated 
    global json_size
    global exceptions
    column_data = df_preprocessed[colmn]
    cardinal = get_cardinality(column_data)
    # print(colmn + ":")
    # print(cardinal)
    column_type = df_preprocessed[colmn].dtype
    column_df = pd.DataFrame(column_data)
    min_value = None

    contains_range = False
    if(column_type == 'int32' or column_type == 'int64' or column_type == 'float32' or column_type == 'float64'):
        contains_range = contains_integer_range(column_data)
    if column_type == 'object':
    # 计算每个字符串的长度
        min_length = df_preprocessed[colmn].str.len().min()

    in_large_integer = colmn in large_integer_column

    #根据基数处理
    if cardinal == 1: 
        json_content = get_json_content("single_value", None, str(column_data[0]))
        # 获取 JSON 文件的大小（统计字典json 文件大小）
        dict_filename = f'json/{file_name}/{file_name}-%s-compress-dict.json' % colmn
        with open(dict_filename, 'w') as json_file:
            json.dump(json_content, json_file)
        json_size = json_size + os.path.getsize(dict_filename)

        #不处理只有一个值的列
        return 
    elif cardinal >  0.5 * len(column_data) or (contains_range):
        # print(colmn, cardinal)
        # print(exceptions)
        process_independence_column(colmn, file_name)
        df_concatenated = pd.concat([df_concatenated, column_df], axis=1)
        return 
# or (pd.api.types.is_numeric_dtype(column_type) and min_value < 10) or (column_type == 'object' and min_length <= 1)


    #字典压缩
    df, dict, top_8_percentage = compress_column(column_data,f'city0/{file_name}-%s-compress.csv'%colmn, colmn)

    
    json_content = get_json_content('dict', dict, None)
    # 获取 JSON 文件的大小（统计字典json 文件大小）
    dict_filename = f'json/{file_name}/{file_name}-%s-compress-dict.json' % colmn
    with open(dict_filename, 'w') as json_file:
        json.dump(json_content, json_file)
    json_size = json_size + os.path.getsize(dict_filename)

    #每一列添加到最后的df数据中
    if df_concatenated.empty:
        df_concatenated = df
    else:
        df_concatenated = pd.concat([df_concatenated, df], axis=1)


def extract_process(column_data, column):
    global df_preprocessed
    column_data = column_data.fillna('')

    # 确保 column_data 是字符串类型
    column_data = column_data.astype(str)

    # 按分号分割数据
    df_split = column_data.str.split(';', expand=True)

    # 创建数值列
    df_values = df_split.applymap(lambda x: x if x else None)
    # print(df_values.dtypes)
    # 创建分号列
    df_separators = df_split.applymap(lambda x: ';' if pd.notna(x) else None)

    # 初始化一个空的DataFrame来存储展开的列
    df_expanded = pd.DataFrame(index=df_values.index)

    # 遍历分割后的列
    for i in range(df_values.shape[1]):
        # 处理数值列
        new_values = []
        for value in df_values.iloc[:, i]:
            try:
                new_values.append(int(value))
            except (ValueError, TypeError):
                new_values.append(None)
        df_expanded[f'{column}_value_{i+1}'] = new_values
        # 处理分号列，注意分号列比数值列少一列，且跳过最后一个分号
        if i < df_separators.shape[1] - 1:
            new_separators = df_separators.iloc[:, i]
            df_expanded[f'{column}_separator_{i+1}'] = new_separators

    # 将展开的列拼接到原数据框中
    if not df_preprocessed.empty:
        df_preprocessed = pd.concat([df_preprocessed, df_expanded], axis=1)
    else:
        df_preprocessed = df_expanded



def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]



if __name__ == "__main__":

    # 示例用法
    csv_files = get_csv_files_without_extension('data/')
    print(csv_files)
    #文件路径
    for file_name in csv_files:
        # print(file_path)
        # file_name = 'city0-4G-1M'
        file_path = f'data/{file_name}.csv' 

        data = pd.read_csv(file_path)
        output_dir = os.path.join(f'json/{file_name}')
        os.makedirs(output_dir, exist_ok=True)
        #大整数的列集合
        large_integer_column = []
        #统计字典的大小
        json_size = 0
        #未处理的列集合
        exceptions = []


        


        #最终df
        df_concatenated  = pd.DataFrame()
        df_preprocessed  = pd.DataFrame()
        df_test  = pd.DataFrame()
        
        pretreatment(data, file_name)
        #分类对列进行字典处理


        # 假设df_preprocessed是你的DataFrame
        columns_info = {col: str(df_preprocessed[col].dtype) for col in df_preprocessed.columns}

        
        columns_list = df_preprocessed.columns.tolist()
        with open(f'json/{file_name}/{file_name}-preprocessed-columns_list.json', 'w') as file:
            json.dump(columns_list, file)
            
        for column_name in df_preprocessed.columns:
            column_data = df_preprocessed[column_name]
            dict_process(column_name, file_name)


        #生成csv文件便于调试
        df_concatenated.to_csv(f'compress_data/csv/{file_name}-compress.csv', index=False)



        # 假设 df_concatenated 是你的 DataFrame
        # 例外的列 (未处理的列）

        # 生成 schema，除了例外的列，其他列都设置为 binary 类型
        fields = []
        # print('exceptions')
        print(exceptions)
        for column in df_concatenated.columns:
            # if column not in exceptions:
                # 保留原来的数据类型
                dtype = pa.binary()
                fields.append(pa.field(column, dtype))
        #     else:
        #         dtype = df_concatenated[column].dtype
        #         if dtype == 'object':
        #             dtype = pa.string()
        #         elif dtype == 'int64':
        #             dtype = pa.int64()
        #         elif dtype == 'float64':
        #             dtype = pa.float64()
        #         elif dtype == 'int32':
        #             dtype = pa.int32()
        #         else:
        #             raise ValueError(f"Unsupported dtype: {dtype}")
        #         fields.append(pa.field(column, dtype))
        # # print(exceptions)
            
        # 创建 schema
        schema = pa.schema(fields)
        # print(schema)
        #存入parquet
        schema = pa.schema([pa.field(column, pa.binary()) for column in df_concatenated.columns])
        table = pa.Table.from_pandas(df_concatenated)
        # print(table)
        pq.write_table(table, 
                       f'compress_data/parquet/{file_name}-compress.parquet', version='2.0', 
                        compression='brotli')
        
        # df_concatenated.to_parquet(f'compress_data/parquet/{file_name}-compress.parquet', index=False)
        data.to_parquet(f'{file_name}.parquet', index=False)

        csv_size = os.path.getsize(f'data/{file_name}.csv')
        
        #没压缩时候的parquet大小
        o_parquet_size = os.path.getsize(f'{file_name}.parquet')
        #压缩后parquet大小
        c_parquet_size = os.path.getsize(f'compress_data/parquet/{file_name}-compress.parquet')



        print(file_name)
        Parquet_compression_ratio = o_parquet_size / csv_size 
        c_Parquet_compression_ratio = c_parquet_size / csv_size 
        print(f"CSV 文件大小: {csv_size} 字节")
        print(f"Parquet 文件大小: {o_parquet_size} 字节")
        print(f"Parquet压缩率: {Parquet_compression_ratio}")
        print(f"压缩后 Parquet 文件大小: {c_parquet_size} 字节")
        print(f"压缩后压到Parquet压缩率: {c_Parquet_compression_ratio}")
        print(f"压缩后字典json大小: {json_size}字节")