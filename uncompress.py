import os
import glob
import lzma
import struct
import pandas as pd
import numpy as np
from bitarray import bitarray
from bitarray.util import int2ba, ba2int

def get_csv_files_without_extension(directory):
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


def load_headers(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def bitarrays_to_dataframe(bitarrays, bit_lengths, column_names, valid_masks):
    df_rows = []
    for idx, ba in enumerate(bitarrays):
        row = []
        pos = 0
        for col_idx, bits in enumerate(bit_lengths):
            if bits == 0:
                # 长度为0列直接返回nan
                row.append(np.nan)
                continue
            chunk = ba[pos:pos+bits]
            mask_chunk = valid_masks[idx][pos:pos+bits]
            # 全为1表示有效，否则为空值
            if len(mask_chunk) == bits and all(mask_chunk):
                # 判定符号（简单判断，大于8位且最高位为1则按有符号处理）
                is_signed = bits > 8 and chunk[0]
                val = ba2int(chunk, signed=is_signed)
                row.append(val)
            else:
                row.append(np.nan)
            pos += bits
        df_rows.append(row)
    return pd.DataFrame(df_rows, columns=column_names)


def decompress(template, bitmaps, differences):
    restored_data = []
    diff_idx = 0
    block_size = len(template) // 8
    total_blocks = len(bitmaps) // block_size
    for i in range(total_blocks):
        ba = bitarray(endian='big')
        for j in range(block_size):
            bit = bitmaps[i * block_size + j]
            if bit == 1:
                ba.extend(template[j*8:(j+1)*8])
            else:
                ba.extend(differences[diff_idx:diff_idx+8])
                diff_idx += 8
        restored_data.append(ba)
    return restored_data


def load_bitarray_from_xz(filepath):
    with lzma.open(filepath, "rb") as f:
        ba = bitarray(endian='big')
        ba.fromfile(f)
    return ba


def load_bit_lengths(filepath):
    with open(filepath, "rb") as f:
        byte_data = f.read()
        return list(struct.unpack(f"{len(byte_data)}B", byte_data))


def load_row_valid_masks(filepath, bits_per_row, num_rows):
    ba = load_bitarray_from_xz(filepath)
    row_masks = [ba[i*bits_per_row:(i+1)*bits_per_row] for i in range(num_rows)]
    return row_masks


if __name__ == "__main__":
    file_names = get_csv_files_without_extension('preprocessed_data')
    for file_name in file_names:
        template = load_bitarray_from_xz(f"template_file/{file_name}_template.bin.xz")
        bitmaps = load_bitarray_from_xz(f"bitmap_file/{file_name}_bitmaps.bin.xz")
        differences = load_bitarray_from_xz(f"difference_file/{file_name}_differences.bin.xz")

        recovered = decompress(template, bitmaps, differences)
        headers = load_headers(f"headers/{file_name}.txt")
        restored_df = bitarrays_to_dataframe(recovered, bit_lengths, headers, row_valid_masks)
        restored_df.to_csv(f'uncompress_file/{("-".join(file_name.split("-")[:-1]))}-uncompress.csv', index=False)
        print("还原DataFrame预览:")
        print(restored_df.head())