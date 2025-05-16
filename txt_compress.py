import os
import glob
import tarfile
import lzma


def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]


if __name__ == '__main__':
    file_names = get_csv_files_without_extension('compress_data/csv/')
    for filename in file_names:
        filename = '-'.join(filename.split('-')[:-1])
        folder_path = f'txt/{filename}'
        archive_name = f'{folder_path}.tar.xz'
        print(f"Processing {folder_path}...")

        try:
            # 删除临时文件和日志文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.tmp') or file.endswith('.log'):
                        os.remove(os.path.join(root, file))
            
            # 使用 tarfile 打包文件夹
            with tarfile.open(f"{folder_path}.tar", 'w') as tar:
                tar.add(folder_path, arcname=os.path.basename(folder_path))

            # 使用 lzma 模块压缩 tar 文件
            with open(f"{folder_path}.tar", 'rb') as f_in:
                with lzma.open(archive_name, 'wb') as f_out:
                    f_out.writelines(f_in)

            # 删除中间的 tar 文件
            os.remove(f"{folder_path}.tar")

            print(f'Successfully compressed {folder_path} to {archive_name}')

        except Exception as e:
            print(f'Error processing {folder_path}: {e}')