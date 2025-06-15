import os
import glob
import pandas as pd
import shutil
import lzma


def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


if __name__ == "__main__":
    df = pd.DataFrame(columns=['id', '1Col Bit', '1Col Diff', '1Col Total', '2Col Bit', '2Col Diff', '2Col Total', 'Metadata', 'Parquet', 'CSV', '1Col+Meta/Parquet', '2Col+Meta/Parquet'])
    file_names = get_csv_files_without_extension('chunks_data/')
    for file_name in file_names:
        id = file_name.split('-')[1]

        # 一次列压缩的xz结果
        single_bitmap_size = os.path.getsize(f'compress_file/single-bitmap-{file_name}.csv.xz')
        single_difference_size = os.path.getsize(f'compress_file/single-difference-{file_name}.csv.xz')
        single_total_size = single_bitmap_size + single_difference_size

        # 二次列压缩的xz结果
        second_bitmap_size = os.path.getsize(f'compress_file/bitmap-difference-{file_name}.csv.xz')
        second_difference_size = os.path.getsize(f'compress_file/difference-difference-{file_name}.csv.xz')
        second_total_size = second_bitmap_size + second_difference_size

        # 二次列+二次行压缩的xz结果
        # row_bitmap_size = os.path.getsize(f'compress_file/bitmap-difference-row-{file_name}.csv.xz')
        # row_difference_size = os.path.getsize(f'compress_file/difference-difference-row-{file_name}.csv.xz')
        # row_total_size = row_bitmap_size + row_difference_size + second_bitmap_size


        if id in df['id'].values:
            # 已有ID：找到对应行并更新size
            df.loc[df['id'] == id, '1Col Bit'] += single_bitmap_size
            df.loc[df['id'] == id, '1Col Diff'] += single_difference_size
            df.loc[df['id'] == id, '1Col Total'] += single_total_size
            df.loc[df['id'] == id, '2Col Bit'] += second_bitmap_size
            df.loc[df['id'] == id, '2Col Diff'] += second_difference_size
            df.loc[df['id'] == id, '2Col Total'] += second_total_size
            # df.loc[df['id'] == id, '2Col+2Row Bit'] += row_bitmap_size
            # df.loc[df['id'] == id, '2Col+2Row Diff'] += row_difference_size
            # df.loc[df['id'] == id, '2Col+2Row Total'] += row_total_size
        else:
            # 新ID：添加新行
            new_row = pd.DataFrame({'id': [id], 
                                    '1Col Bit': [single_bitmap_size], '1Col Diff': [single_difference_size], '1Col Total': [single_total_size],
                                   '2Col Bit': [second_bitmap_size], 
                                   '2Col Diff': [second_difference_size], 
                                   '2Col Total': [second_total_size],
                                #    '2Col+2Row Bit': [row_bitmap_size], 
                                #    '2Col+2Row Diff': [row_difference_size], 
                                #    '2Col+2Row Total': [row_total_size],
                                #    '1Col Throughput':[None],
                                #    '2Col Throughput':[None],
                                   'Metadata': [None], 
                                   'Parquet': [None], 
                                   'CSV': [None],
                                   '1Col+Meta/Parquet': [None], 
                                   '2Col+Meta/Parquet': [None], 
                                #    '2Col+2Row+Meta/Parquet': [None],
                                   })
            df = pd.concat([df, new_row], ignore_index=True)


    file_names = get_csv_files_without_extension('data/')
    for file_name in file_names:
        # 元数据大小
        metadata_size = os.path.getsize(f'txt/dict-{file_name}.tar.xz')

        # pq大小
        parquet_df = pd.read_csv('parquet_size.csv', dtype=str)
        print(file_name)
        parquet_size = int(parquet_df[parquet_df['File Name'] == file_name]['Parquet Size'].values[0])

        # 吞吐率
        # time_1_df = pd.read_csv('first_compress.csv')
        # # 使用 str.contains() 筛选包含 file_name 的行
        # time_1 = time_1_df[time_1_df['File Name'].str.contains(file_name)]['Template1 Time'].sum()

        # time_2_df = pd.read_csv('twice_compress.csv')
        # # 使用 str.contains() 筛选包含 file_name 的行
        # time_2 = time_2_df[time_2_df['File Name'].str.contains(file_name)]['Template2 Time'].sum()

        csv_size = os.path.getsize(f'data/{file_name}.csv')
        # throughput_1 = csv_size / time_1
        # throughput_2 = csv_size / time_2

        # 获取对应 id 的行（假设每行的 id 是唯一的）
        row = df.loc[df['id'] == file_name].iloc[0]
        
        # 进行计算，确保各列都是数值类型
        single_ratio = (float(row['1Col Total']) + metadata_size) / parquet_size
        second_ratio = (float(row['2Col Total']) + metadata_size) / parquet_size
        # row_ratio = (float(row['2Col+2Row Total']) + metadata_size) / parquet_size
        
        # print(file_name)
        # print(row)
        # print(throughput_1,throughput_2,single_ratio,second_ratio,row_ratio)
        # df.loc[df['id'] == file_name, '1Col Throughput'] = throughput_1
        # df.loc[df['id'] == file_name, '2Col Throughput'] = throughput_2
        df.loc[df['id'] == file_name, 'Metadata'] = metadata_size
        df.loc[df['id'] == file_name, 'Parquet'] = parquet_size
        df.loc[df['id'] == file_name, 'CSV'] = csv_size
        df.loc[df['id'] == file_name, '1Col+Meta/Parquet'] = single_ratio
        df.loc[df['id'] == file_name, '2Col+Meta/Parquet'] = second_ratio
        # df.loc[df['id'] == file_name, '2Col+2Row+Meta/Parquet'] = row_ratio

    # 在所有数据处理完成后，添加汇总行
    total_row = df.drop(columns='id').sum().to_dict()
    total_row['id'] = 'Total'

    # 计算特殊列的值
    total_metadata = df['Metadata'].sum()
    total_parquet = df['Parquet'].sum()

    # 使用公式重新计算特殊列
    total_row['1Col+Meta/Parquet'] = (df['1Col Total'].sum() + total_metadata) / total_parquet
    total_row['2Col+Meta/Parquet'] = (df['2Col Total'].sum() + total_metadata) / total_parquet
    # total_row['2Col+2Row+Meta/Parquet'] = (df['2Col+2Row Total'].sum() + total_metadata) / total_parquet

    # 将 Throughput 列改为计算平均值
    # total_row['1Col Throughput'] = df['1Col Throughput'].mean()
    # total_row['2Col Throughput'] = df['2Col Throughput'].mean()

    # 将汇总行添加到 DataFrame 末尾
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    # 将结果保存到 CSV
    df.to_csv('lossless-result.csv', index=False)
