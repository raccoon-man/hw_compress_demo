import pandas as pd
import lzma
import os
import glob


def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


def is_all_zeros(s):
    return set(s) == {'1'}

def uncompress_xz_to_csv(xz_file_path, csv_file_path):
    try:
        with lzma.open(xz_file_path, 'rt', encoding='utf-8') as xz_file:
            with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
                csv_file.write(xz_file.read())
        print(f"成功将 {xz_file_path} 解压为 {csv_file_path}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {xz_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    file_names = get_csv_files_without_extension('chunks_data/')

    for filename in file_names:
        headers = []
        # 以读取模式打开文件
        with open(f'headers/{filename}.txt', 'r') as file:
            for line in file:
                # 去除每行末尾的换行符并添加到列表
                headers.append(line.strip())


        # 读取Template，为list
        templates1 = pd.read_csv(f'template_file/single-{filename}.csv',header=None,dtype=str, skip_blank_lines=False)
        
        uncompress_xz_to_csv(f'compress_file/single-difference-{filename}.csv.xz', f'compress_file/single-difference-{filename}.csv')
        uncompress_xz_to_csv(f'compress_file/single-bitmap-{filename}.csv.xz', f'compress_file/single-bitmap-{filename}.csv')
        
        # 读取压缩后数据，按照从左至右遍历解压
        bitmaps = pd.read_csv(f'compress_file/single-bitmap-{filename}.csv',header=None,dtype=str, skip_blank_lines=False)

        differences = pd.read_csv(f'compress_file/single-difference-{filename}.csv',header=None,dtype=str, skip_blank_lines=False)

        col_number = bitmaps.shape[1]
        row_number = bitmaps.shape[0]

        uncompress_col_number = 0
        for elem in templates1.iloc[0,:]:
            if pd.isna(elem):
                elem = ''
            lst = str(elem).split(',')
            uncompress_col_number += len(lst)
        
        # 创建一个 row 行 uncompress_col_number 列的空 DataFrame
        df = pd.DataFrame(index=range(row_number), columns=range(uncompress_col_number - 1))
        uncompress_i = 0
        uncompress_j = 0
        
        for i in range(0, row_number):
            for j in range(0, col_number):
                temp_i = i // 50
                temp_j = j
                template1 = templates1.iloc[temp_i, temp_j]
                hex_string = str(bitmaps.iloc[i,j])
                binary_string = ''
                for char in hex_string:
                    binary_digit = bin(int(char, 16))[2:].zfill(4)
                    binary_string += binary_digit
                bitmap = ''
                for bin_idx in range(0, len(binary_string), 2):
                    if binary_string[bin_idx:bin_idx+2] == '00':
                        bitmap = bitmap + '0'
                    elif binary_string[bin_idx:bin_idx+2] == '01':
                        bitmap = bitmap + '1'
                    elif binary_string[bin_idx:bin_idx+2] == '10':
                        bitmap = bitmap + ' '
                difference = str(differences.iloc[i,j])
                uncompressed_data = ''

                # temp_idx = 0
                # diff_idx = 0
                # for cur in range(0, len(bitmap)):
                #     if bitmap[cur] == ' ':
                #         uncompressed_data += ' '
                #     elif bitmap[cur] == '1':
                #         uncompressed_data += template1[cur]
                #     elif bitmap[cur] == '0':
                #         uncompressed_data += difference[diff_idx]
                #         diff_idx += 1

                bit_idx = 0
                diff_idx = 0
                for temp in template1:
                    if temp == ',':
                        uncompressed_data += ','
                    else:
                        if bitmap[bit_idx] == ' ':
                            uncompressed_data += ' '
                        elif bitmap[bit_idx] == '1':
                            uncompressed_data += temp
                        elif bitmap[bit_idx] == '0':
                            uncompressed_data += difference[diff_idx]
                            diff_idx += 1
                        bit_idx += 1
                uncompressed_list = uncompressed_data.split(',')

                for elem in uncompressed_list:
                    if pd.isna(elem):
                        elem = ''
                    elem = str(elem).strip(' ')
                    df.at[uncompress_i, uncompress_j] = elem
                    uncompress_j += 1
                    if uncompress_j == uncompress_col_number:
                        uncompress_j = 0
                        uncompress_i += 1
            
        df.columns = headers
        df.to_csv(f'uncompress_file/{filename}.csv',index=False)
            



        
    


