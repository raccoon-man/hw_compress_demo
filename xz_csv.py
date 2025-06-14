import os
import glob
import lzma
import shutil
import pandas as pd

def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]

def compress_csv_with_xz(input_file, preset=6):
    """压缩CSV文件为xz格式并返回压缩后的文件大小"""
    try:
        output_file = f"{input_file}.xz"
        with open(input_file, 'rb') as f_in:
            with lzma.open(output_file, 'wb', preset=preset) as f_out:
                shutil.copyfileobj(f_in, f_out)
        return os.path.getsize(output_file)
    except Exception as e:
        print(f"压缩文件 {input_file} 时出错: {e}")
        return None

if __name__ == "__main__":
    directory = 'data/'
    file_names = get_csv_files_without_extension(directory)
    
    # 读取主DataFrame
    df = pd.read_csv('city1-5G-无损.csv')
    
    # 创建ID到压缩文件大小的映射
    id_to_size = {}
    for id in file_names:
        csv_path = f'{directory}{id}.csv'
        xz_size = compress_csv_with_xz(csv_path)
        print(id,xz_size)
        if xz_size is not None:
            id_to_size[id] = xz_size
            print(f"文件 {csv_path} 压缩成功，大小: {xz_size} 字节")
    
    # 在DataFrame中添加xz_size列，根据ID匹配
    df['xz_size'] = df['ID'].map(id_to_size)
    
    # 保存结果
    df.to_csv('city1-5G-无损_带大小.csv', index=False)
    print(f"处理完成，结果已保存到 city1-5G-无损_带大小.csv")