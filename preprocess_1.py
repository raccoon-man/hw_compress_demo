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
from dict import get_csv_files_without_extension, save_df_metadata,convert_to_nullable_int,remove_prefix,median_cardinality,dict_process
from dict import save_custom_txt_from_json,convert_keys_to_str, pretreatment


if __name__ == "__main__":
    # 设置全局参数，表示进行列压缩的次数


        csv_files = get_csv_files_without_extension('data/')
        
        print(csv_files)
        dimension = {}
        dimension['16843010'] = 'body_1005'
        dimension['16777219'] = 'head_12'
        dimension['16777220'] = 'head_12'
        
        for file_name in csv_files:

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
            
            total_json_str_keys = convert_keys_to_str(total_json)
            save_custom_txt_from_json(total_json_str_keys, file_name)