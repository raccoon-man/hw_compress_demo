import pandas as pd
import os
import glob
import struct
import lzma
import shutil
from collections import Counter
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import numpy as np


def get_csv_files_without_extension(directory):
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


def determine_min_bits(series):
    min_val = series.min()
    max_val = series.max()
    if min_val < 0:
        bits = 8
        while True:
            if bits == 8 and -128 <= min_val and max_val <= 127:
                return bits
            elif bits == 16 and -32768 <= min_val and max_val <= 32767:
                return bits
            elif bits == 32:
                return bits
            bits *= 2
    else:
        bits = 8
        while True:
            if bits == 8 and max_val <= 255:
                return bits
            elif bits == 16 and max_val <= 65535:
                return bits
            elif bits == 32:
                return bits
            bits *= 2


def convert_columns_to_bitarrays(df):
    column_bitarrays = []
    bit_lengths = []
    valid_masks = []

    for col in df.columns:
        series = df[col]
        non_null_series = series[series.notnull()]

        # 判断非空部分是否全为整数（允许浮点但值为整数）
        def is_all_int(s):
            try:
                return all(float(x).is_integer() for x in s)
            except:
                return False

        if len(non_null_series) == 0:
            # 全空列，bit长度记0
            bit_lengths.append(0)
            column_bitarrays.append([bitarray() for _ in range(len(df))])
            valid_masks.append([bitarray() for _ in range(len(df))])
            continue

        if not is_all_int(non_null_series):
            # 非整数列，不转成bitarray，直接用长度0占位（可根据需求调整）
            bit_lengths.append(0)
            column_bitarrays.append([bitarray() for _ in range(len(df))])
            valid_masks.append([bitarray() for _ in range(len(df))])
            continue

        int_series = non_null_series.astype(int)
        bits = determine_min_bits(int_series)
        bit_lengths.append(bits)
        is_signed = int_series.min() < 0

        ba_list = []
        mask_list = []
        for val in series:
            if pd.isnull(val):
                ba_list.append(bitarray('0' * bits))
                mask_list.append(bitarray('0' * bits))
            else:
                ba_val = int2ba(int(val), length=bits, signed=is_signed, endian='big')
                ba_list.append(ba_val)
                mask_list.append(bitarray('1' * bits))

        column_bitarrays.append(ba_list)
        valid_masks.append(mask_list)

    # 按行拼接bitarray和掩码
    row_bitarrays = []
    row_masks = []
    for row_idx in range(len(df)):
        row_bits = bitarray(endian='big')
        row_mask = bitarray(endian='big')
        for col_idx in range(len(column_bitarrays)):
            row_bits.extend(column_bitarrays[col_idx][row_idx])
            row_mask.extend(valid_masks[col_idx][row_idx])
        row_bitarrays.append(row_bits)
        row_masks.append(row_mask)

    return row_bitarrays, bit_lengths, row_masks


def find_template_bitarray(bitarrays):
    length = len(bitarrays[0])
    template = bitarray(length)
    for i in range(length):
        bits = [ba[i] for ba in bitarrays]
        common = Counter(bits).most_common(1)[0][0]
        template[i] = common
    return template


def compress(template, data_list):
    bitmaps = bitarray(endian='big')
    differences = bitarray(endian='big')

    block_size = len(template) // 8
    for elem in data_list:
        res = elem ^ template
        bitmap = bitarray(endian='big')
        difference = bitarray(endian='big')
        for i in range(0, len(res), 8):
            block = res[i:i+8]
            if block.count() == 0:
                bitmap.append(1)
            else:
                bitmap.append(0)
                difference.extend(elem[i:i+8])
        bitmaps.extend(bitmap)
        differences.extend(difference)
    return bitmaps, differences



def save_bitarray_to_file(bitarr, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        bitarr.tofile(f)


def save_bit_lengths(bit_lengths, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        for bits in bit_lengths:
            f.write(struct.pack("B", bits))


def save_headers(columns, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for col in columns:
            f.write(col + "\n")


def save_row_valid_masks(row_masks, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    combined = bitarray(endian='big')
    for mask in row_masks:
        combined.extend(mask)
    with open(filepath, "wb") as f:
        combined.tofile(f)


def compress_file_with_xz(input_file, preset):
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


if __name__ == "__main__":
    file_names = get_csv_files_without_extension('preprocessed_data')
    for file_name in file_names:
        print(f"正在处理文件：{file_name}.csv")
        df = pd.read_csv(f'preprocessed_data/{file_name}.csv')
        row_bitarrays, bit_lengths, row_valid_masks = convert_columns_to_bitarrays(df)
        save_headers(df.columns, f"headers/{file_name}.txt")
        
        template = find_template_bitarray(row_bitarrays)
        bitmaps, differences = compress(template, row_bitarrays)

        save_bit_lengths(bit_lengths, f"bit_length_file/{file_name}_bitlengths.bin")
        save_bitarray_to_file(bitmaps, f"bitmap_file/{file_name}_bitmaps.bin")
        save_bitarray_to_file(differences, f"difference_file/{file_name}_differences.bin")
        save_bitarray_to_file(template, f"template_file/{file_name}_template.bin")
        save_row_valid_masks(row_valid_masks, f"valid_mask_file/{file_name}_rowmask.bin")

        compress_file_with_xz(f"valid_mask_file/{file_name}_rowmask.bin", 6)
        compress_file_with_xz(f"bit_length_file/{file_name}_bitlengths.bin",6)
        compress_file_with_xz(f"bitmap_file/{file_name}_bitmaps.bin",6)
        compress_file_with_xz(f"difference_file/{file_name}_differences.bin",6)
        compress_file_with_xz(f"template_file/{file_name}_template.bin",6)

        
