import pandas as pd
import numpy as np
import compress2
import math
import time
import warnings
import shutil
import lzma
import os
import zstd
import glob
import csv

def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]

warnings.filterwarnings('ignore')


def compress_and_save_string(input_string, file_path):
    try:
        # 将字符串编码为字节类型，因为 zstd 处理字节
        original_bytes = input_string.encode('utf-8')

        # 压缩字节数据
        compressed = zstd.compress(original_bytes)

        # 保存压缩后的字节数据到二进制文件
        with open(file_path, 'wb') as file:
            file.write(compressed)
        print(f"压缩数据已保存到 {file_path}")

        # 从二进制文件读取压缩数据
        with open(file_path, 'rb') as file:
            read_compressed = file.read()

        # 解压缩字节数据
        decompressed_bytes = zstd.decompress(read_compressed)

        # 将解压缩后的字节数据转换回字符串
        decompressed_string = decompressed_bytes.decode('utf-8')

        return decompressed_string
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def compress_csv_with_xz(input_file, preset):
    try:
        # 打开输入文件和输出文件
        with open(input_file, 'rb') as f_in:
            output_file = f"{input_file}.xz"
            with lzma.open(output_file, 'wb', preset=preset) as f_out:
                # 压缩文件内容
                shutil.copyfileobj(f_in, f_out)

        # print(f"文件 {input_file} 压缩成功！")
    except Exception as e:
        print(f"压缩文件时出错: {e}")


# 以byte为单位找到众数
def find_majority_per_position(lst):
    if len(lst) == 0:
        n = 0
    else:
        n = len(lst[0])
    majority_list = []
    for i in range(n):
        column = [sub_lst[i] for sub_lst in lst]
        counts = {}
        for item in column:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        # 排除模板中的填充符
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if sorted_counts[0][0] == '00100000' and len(sorted_counts) > 1:
            majority_list.append(sorted_counts[1][0])
        else:
            majority_list.append(sorted_counts[0][0])
    return majority_list

# 把数据切分为50*5的块
def split_data(data, r, c):
    block_list = []
    for start_row in range(0, data.shape[0], r):
        end_row = start_row + r
        if end_row > data.shape[0]:
            end_row = data.shape[0]
        for start_col in range(0, data.shape[1], c):
            end_col = start_col + c
            if end_col > data.shape[1]:
                end_col = data.shape[1]
            df = data.iloc[start_row:end_row, start_col:end_col]
            df.reset_index(drop=True, inplace=True)
            block_list.append(df)
    return block_list

# 计算两个二进制list的相似度
def count_similarity(cur_list, template_list):
    same_byte = 0
    for byte_1, byte_2 in zip(cur_list, template_list):
        if byte_1 == byte_2:
            same_byte += 1
    return same_byte/len(cur_list)

# 每块提取前k个众数模板
def find_template(block_list):
    template_list = []
    for df in block_list:
        
        # 对每行数据的所有数据进行横向拼接，分隔符为','
        combined_list = df.agg(','.join, axis = 1)
        # 将拼接后的字符串以byte为单位转换为二进制列表
        binary_lists = combined_list.apply(lambda x: [bin(ord(c))[2:].zfill(8) for c in x]).tolist()

        majority_list = find_majority_per_position(binary_lists)

        ascii_string = ''.join([chr(int(binary, 2)) for binary in majority_list])

        template_list.append(ascii_string)

    return template_list


def calculate_similarity(str1, str2):
    diff_cnt = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    return 1 - diff_cnt / len(str1)

def calculate_similarity_block(str, block):
    similarity = []
    for i in range(0, len(block)):
        similarity.append(calculate_similarity(str, block.loc[i]))
    sim = sum(similarity) / len(similarity)
    return sim

# 模板压缩还原
# 还原block_list为原始数据框
def restore_data(block_list, original_shape):
    rows, cols = original_shape
    # 初始化一个空的数据框
    restored_data = pd.DataFrame(np.nan, index=range(rows), columns=range(cols))
    block_index = 0
    for start_row in range(0, rows, 50):
        end_row = start_row + 50
        if end_row > rows:
            end_row = rows
        for start_col in range(0, cols, 5):
            end_col = start_col + 5
            if end_col > cols:
                end_col = cols
            # 获取当前块
            block = block_list[block_index]
            # 将块的数据赋值给还原后的数据框
            restored_data.iloc[start_row:end_row, start_col:end_col] = block.values
            block_index += 1
    return restored_data


def align_lengths(str1, data):
    # 如果 data 是 numpy 数组，将其转换为字典
    if isinstance(data, np.ndarray):
        num_cols = data.shape[1]
        data = {f'col{i+1}': data[:, i].tolist() for i in range(num_cols)}

    # 将 str1 按逗号分割成列表
    str1_list = str1.split(',')
    # 获取每个位置的最大长度
    max_lengths = []
    num_items = len(str1_list)
    for i in range(num_items):
        # 收集每个位置的所有元素
        values = [str1_list[i]]
        for val in data.iloc[:, i]:
            values.append(val)
        # 找到该位置的最大长度
        max_length = max(len(val) for val in values)
        max_lengths.append(max_length)

    # 对 str1 进行填充
    new_str1_list = []
    for i, item in enumerate(str1_list):
        new_str1_list.append(item.ljust(max_lengths[i], ' '))
    new_str1 = ','.join(new_str1_list)

    # 对 data 中的每个列进行填充
    new_data = {}
    i = 0
    for col_name, col_values in data.items():
        new_col_values = []
        for value in col_values:
            new_col_values.append(value.ljust(max_lengths[i], ' '))
        new_data[col_name] = new_col_values
        i += 1
    new_data = pd.DataFrame(new_data)
    return new_str1, new_data


def csv_to_df_list(file_path, encoding='utf-8'):
    data = []
    with open(file_path, 'r', newline='', encoding=encoding) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # template_time = []
    file_names = get_csv_files_without_extension('chunks_data/')
    
    for filename in file_names:
        print(filename)
        filename = 'difference-' + filename
        df = pd.read_csv(f'compress_file/{filename}.csv',header=None,dtype=str, skip_blank_lines=False).astype(str).replace('nan','')

        # print(df.head())
        # for col in df.columns:
        #     # 获取最大字符长度
        #     max_length = df[col].str.len().max()
        #     # 补全前导+
        #     if max_length > 0:
        #         df[col] = df[col].str.ljust(max_length, ' ')

        total_size = 0
        compressed_data = []
        difference = []
        templates = []
        len_data = []
        # time_1 = 0
        # time_2 = 0
        # time_3 = 0

        # 读取前一次模板
        new_filename = '-'.join(filename.split('-')[1:])
        template1s = pd.read_csv(f'template_file/{new_filename}.csv', skip_blank_lines=False,dtype=str,header=None).astype(str).replace('nan','')
        

        df_original = pd.read_csv(f'chunks_data/{new_filename}.csv', skip_blank_lines=False,dtype=str).astype(str).replace('nan','')
        
        # print(df.head())
        # for col in df_original.columns:
        #     # 获取最大字符长度
        #     max_length = df_original[col].str.len().max()
        #     # 补全前导+
        #     if max_length > 0:
        #         df_original[col] = df_original[col].str.ljust(max_length, ' ')

        for i in range(0, len(df_original.columns), 1):
            # print(i)
            data = df.iloc[:, i:i + 1]
            # print(data)
            data2 = df_original.iloc[:, i*len(df_original.columns):(i + 1)*len(df_original.columns)]
            block_list = split_data(data, 50, 1)
            block_list_original = split_data(data2, 50, len(df_original.columns))

            
            template_row = 0
            for block, block_original in zip(block_list, block_list_original):
                
                for col in block.columns:
                    # 获取最大字符长度
                    max_length = block[col].str.len().max()
                    # 补全前导+
                    if max_length > 0:
                        block[col] = block[col].str.ljust(max_length, ' ')

                for col in block_original.columns:
                    # 获取最大字符长度
                    max_length = block_original[col].str.len().max()
                    # 补全前导+
                    if max_length > 0:
                        block_original[col] = block_original[col].str.ljust(max_length, ' ')
                if template_row == len(template1s):
                    template_row = 0
                template_final = ''
                template1 = str(template1s.iloc[template_row,i])
                max_similarity = -1
                # start_time = time.time()
                # 对每行数据的所有数据进行横向拼接，分隔符为','
                combined_list = block.agg(','.join, axis = 1)
                # 将拼接后的字符串以byte为单位转换为二进制列表
                binary_lists = combined_list.apply(lambda x: [bin(ord(c))[2:].zfill(8) for c in x]).tolist()
                # print(binary_lists)
                majority_list = find_majority_per_position(binary_lists)
                template_final = ''.join([chr(int(binary, 2)) for binary in majority_list])
                # end_time = time.time()
                # time_2 += (end_time - start_time)
                # print(template_final)

                # start_time = time.time()
                dt, diff = compress2.compress_block(template1, template_final, block_original)
                # end_time = time.time()

                # time_3 += (end_time - start_time)

                compressed_data += dt
                difference += diff
                templates.append(template_final)
                len_data.append(len(template_final))
                template_row += 1


        col_number = math.ceil(df_original.shape[1] / len(df_original.columns))
        row_number = df_original.shape[0]

        # 打印生成模板，匹配模板，模板压缩的吞吐量
        # template_time.append(time_3)


        # 创建一个 row 行 col 列的空 DataFrame
        df = pd.DataFrame(index=range(row_number), columns=range(col_number))

        # # 列表索引
        # index = 0
        # # 遍历列
        # for col in df.columns:
        #     # 遍历行
        #     for row in df.index:
        #         if index < len(compressed_data):
        #             df.at[row, col] = compressed_data[index]
        #             index += 1
        #         else:
        #             break
        #     if index >= len(compressed_data):
        #         break
                
        # df.astype(str).to_csv(f'compress_file/bitmap-{filename}.csv',index=False,header=False)
        # compress_csv_with_xz(f'compress_file/bitmap-{filename}.csv', 6)

        # 列表索引
        index = 0
        # 遍历列
        for col in df.columns:
            # 遍历行
            for row in df.index:
                if index < len(compressed_data):
                    bitmap = ''
                    for elem in compressed_data[index]:
                        if elem == '0':
                            bitmap += '00'
                        elif elem == '1':
                            bitmap += '01'
                        elif elem == ' ':
                            bitmap += '10'
                        elif elem == '2':
                            bitmap += '11'
                    while len(bitmap) % 4 != 0:
                        bitmap += '10'
                    # 将 bitmap 转换为十六进制字符
                    hex_string = ''
                    for i in range(0, len(bitmap), 4):
                        hex_digit = int(bitmap[i:i + 4], 2)
                        hex_string += hex(hex_digit)[2:].zfill(1)
                    df.at[row, col] = str(hex_string)
                    index += 1
                else:
                    break
            if index >= len(compressed_data):
                break 
        df.astype(str).to_csv(f'compress_file/bitmap-{filename}.csv',index=False,header=False)
        compress_csv_with_xz(f'compress_file/bitmap-{filename}.csv', 6)
        
        df = pd.DataFrame(index=range(row_number), columns=range(col_number))

        # 列表索引
        index = 0
        # 遍历列
        for col in df.columns:
            # 遍历行
            for row in df.index:
                if index < len(difference):
                    df.at[row, col] = difference[index]
                    index += 1
                else:
                    break
            if index >= len(difference):
                break

        df.astype(str).to_csv(f'compress_file/difference-{filename}.csv',index=False,header=False)
        compress_csv_with_xz(f'compress_file/difference-{filename}.csv',6)

        row_number = math.ceil(df.shape[0] / 50)


        # 创建一个 row 行 col 列的空 DataFrame
        df = pd.DataFrame(index=range(row_number), columns=range(col_number))
        # print(templates)
        # 列表索引
        index = 0
        # 遍历列
        for col in df.columns:
            # 遍历行
            for row in df.index:
                if index < len(templates):
                    df.at[row, col] = templates[index]
                    index += 1
                else:
                    break
            if index >= len(templates):
                break

        df.astype(str).to_csv(f'template_file/{filename}.csv',index=False,header=False)
    # data = {'File Name': file_names, 'Template2 Time': template_time}
    # df = pd.DataFrame(data)
    # df.to_csv('twice_compress.csv', index=False)
    
