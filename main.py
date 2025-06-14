import subprocess
import os
import time
import pandas as pd
import json


# 解析参数

# Compress_Chunk_Size代表每次读入内存和提取一条模板的行数
Compress_Chunk_Size = 1000

# 数据文件夹路径
DATA_FOLDER = 'data/'

commands = [
    "python clear_folder.py",
    "python transformation_format.py",
    "python dict.py",
    f"python column_compress.py --compress-chunk-size {Compress_Chunk_Size}",
    # f"python column_uncompress.py --compress-chunk-size {Compress_Chunk_Size}", # 解压缩
    # f"python compare.py --compress-chunk-size {Compress_Chunk_Size}", # 解压缩
    "python txt_compress.py",
    f"python count_size.py",
]

# 设置环境变量以强制无缓冲输出
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'


# 计算文件夹大小(MB)
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # 转换为MB

# 存储每个脚本的执行时间
execution_times = {}
total_time = 0.0


for command in commands:
    try:
        print(f"正在执行命令: {command}")

        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        stderr = process.stderr.read()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, output=stderr)

        print("命令执行成功")
        
            
    except subprocess.CalledProcessError as e:
        print(f"命令 {command} 执行失败，错误信息如下:")
        print(e.stderr)
        exit(1)
