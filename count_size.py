import os
import glob
import pandas as pd
import lzma
import shutil
import argparse
import json


def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


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


if __name__ == "__main__":
    
    file_names = get_csv_files_without_extension('compress_data/csv')


    result_df = pd.DataFrame({
    'ID': [filename.split('-')[1] for filename in file_names]
    })
    
    bitmap_sizes1 = []
    bitmap_sizes2 = []
    difference_sizes = []
    total_sizes = []
    res = {}
    cr_map = {'c':'Col','r':'Row'}

    for file_name in file_names:

        if '16777219' in file_name:
            Operation = ['c', 6, 'r', 0]
        else:
            Operation = ['c', 1, 'r', 0]

        id = file_name.split('-')[1]
        if Operation[3] > 0:

            # 第一层压缩的CSV路径
            bitmap_path1 = f'bitmap_file/bitmap-{Operation[0]}{Operation[1]}-{file_name}.csv'
            bitmap_path2 = f'bitmap_file/bitmap-{Operation[2]}{Operation[3]}-{file_name}.csv'
            difference_path2 = f'difference_file/difference-{Operation[2]}{Operation[3]}-{file_name}.csv'

            # 第一层压缩的XZ路径
            bitmap_xz_path1 = f'bitmap_file/bitmap-{Operation[0]}{Operation[1]}-{file_name}.csv.xz'
            bitmap_xz_path2 = f'bitmap_file/bitmap-{Operation[2]}{Operation[3]}-{file_name}.csv.xz'
            difference_xz_path2 = f'difference_file/difference-{Operation[2]}{Operation[3]}-{file_name}.csv.xz'

            # XZ压缩
            compress_csv_with_xz(bitmap_path1,6)
            compress_csv_with_xz(bitmap_path2,6)
            compress_csv_with_xz(difference_path2,6)

            # 获取XZ大小
            bitmap_size1 = os.path.getsize(bitmap_xz_path1)
            bitmap_size2 = os.path.getsize(bitmap_xz_path2)
            difference_size = os.path.getsize(difference_xz_path2)
            total_size = bitmap_size1 + bitmap_size2 + difference_size

            bitmap_sizes1.append(bitmap_size1)
            bitmap_sizes2.append(bitmap_size2)
            difference_sizes.append(difference_size)
            total_sizes.append(total_size)
            
        else:
            # 第一层压缩的CSV路径
            bitmap_path1 = f'bitmap_file/bitmap-{Operation[0]}{Operation[1]}-{file_name}.csv'
            difference_path1 = f'difference_file/difference-{Operation[0]}{Operation[1]}-{file_name}.csv'
            # 第一层压缩的XZ路径
            bitmap_xz_path1 = f'bitmap_file/bitmap-{Operation[0]}{Operation[1]}-{file_name}.csv.xz'
            difference_xz_path1 = f'difference_file/difference-{Operation[0]}{Operation[1]}-{file_name}.csv.xz'
            # XZ压缩
            compress_csv_with_xz(bitmap_path1,6)
            compress_csv_with_xz(difference_path1,6)
            # 获取XZ大小
            bitmap_size = os.path.getsize(bitmap_xz_path1)
            difference_size = os.path.getsize(difference_xz_path1)
            total_size = bitmap_size + difference_size

            bitmap_sizes1.append(bitmap_size)
            difference_sizes.append(difference_size)
            total_sizes.append(total_size)
            

    if Operation[3] > 0:
        df = pd.DataFrame({
                f'{Operation[1]}{cr_map[Operation[0]]} Bitmap':bitmap_sizes1,
                f'{Operation[3]}{cr_map[Operation[2]]} Bitmap':bitmap_sizes2,
                f'{Operation[1]}{cr_map[Operation[0]]} Difference':difference_sizes,
                f'{Operation[1]}{cr_map[Operation[0]]}+{Operation[3]}{cr_map[Operation[2]]} Total':total_sizes,
            })
        result_df = pd.concat([result_df, df], axis=1)
    else:
        df = pd.DataFrame({
                f'{Operation[1]}{cr_map[Operation[0]]} Bitmap':bitmap_sizes1,
                f'{Operation[1]}{cr_map[Operation[0]]} Difference':difference_sizes,
                f'{Operation[1]}{cr_map[Operation[0]]} Total':total_sizes,
            })
        result_df = pd.concat([result_df, df], axis=1)
        
    metadata_sizes = []
    parquet_sizes = []
    csv_sizes = []

    for file_name in file_names:
        txt_name = '-'.join(file_name.split('-')[:-1])
        txt_path = f'txt/{txt_name}.tar.xz'
        txt_size = os.path.getsize(txt_path)
        metadata_sizes.append(txt_size)

        parquet_df = pd.read_csv('parquet_size.csv', dtype=str)
        id = file_name.split('-')[1]
        print(id)
        parquet_size = int(parquet_df[parquet_df['File Name'] == id]['Parquet Size'].values[0])
        parquet_sizes.append(parquet_size)

        csv_path = f'data/{id}.csv'
        csv_size = os.path.getsize(csv_path)
        csv_sizes.append(csv_size)


    meta_df = pd.DataFrame({
        'Metadata':metadata_sizes,
        'Parquet':parquet_sizes,
        'CSV':csv_sizes
    })

    result_df = pd.concat([result_df, meta_df], axis=1)
    # 计算汇总行（ID列设置为"Total"，其他列计算总和）
    summary_row = pd.DataFrame({
        'ID': ['Total'],
        **{col: [result_df[col].sum()] for col in result_df.columns if col != 'ID'}
    })
    # 将汇总行添加到DataFrame底部
    result_df = pd.concat([result_df, summary_row], ignore_index=True)

    if Operation[3] > 0:
        result_df['Total/Parquet'] = (result_df[f'{Operation[1]}{cr_map[Operation[0]]}+{Operation[3]}{cr_map[Operation[2]]} Total']+result_df['Metadata'])/result_df['Parquet']
    else:
        result_df['Total/Parquet'] = (result_df[f'{Operation[1]}{cr_map[Operation[0]]} Total']+result_df['Metadata'])/result_df['Parquet']
    
    result_df['Total/CSV'] = (result_df[f'{Operation[1]}{cr_map[Operation[0]]} Total']+result_df['Metadata'])/result_df['CSV']

    result_df['Parquet/CSV'] = result_df['Parquet']/result_df['CSV']


    result_df.to_csv(f"lossless-result.csv",index=False)



            

