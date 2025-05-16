import subprocess
import os

commands = [
    "python clear_folder.py",
    "python transformation_format.py",
    "python dict.py",
    "python row_slicer.py",
    "python single_compress.py",
    "python single_uncompress.py", # 解压缩
    "python compare.py", # 解压缩
    "python first_column_compress.py",
    "python second_column_compress.py",
    "python two_column_uncompress.py", # 解压缩
    "python compare.py", # 解压缩
    "python txt_compress.py",
    "python count_size.py",
    "python dict_uncompress.py" # 解压缩
]

# 设置环境变量以强制无缓冲输出
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'

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
    