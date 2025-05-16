import csv
from io import StringIO
import os
import glob


def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]

def compare_csv_files(file1_str, file2_str):
    reader1 = csv.reader(StringIO(file1_str))
    reader2 = csv.reader(StringIO(file2_str))
    row_num = 0
    has_difference = False
    # 逐行比较
    for row1, row2 in zip(reader1, reader2):
        row_num += 1
        for col_num, (cell1, cell2) in enumerate(zip(row1, row2)):
            try:
                val1 = float(cell1)
                val2 = float(cell2)
                if val1 != val2:
                    has_difference = True
                    print(f"第 {row_num} 行，第 {col_num + 1} 列不同: {cell1} != {cell2}")
            except (ValueError, TypeError):
                if cell1 != cell2:
                    has_difference = True
                    print(f"第 {row_num} 行，第 {col_num + 1} 列不同: {cell1} != {cell2}")

    # 检查两个文件的行数是否相同
    remaining_rows1 = list(reader1)
    remaining_rows2 = list(reader2)
    if remaining_rows1:
        has_difference = True
        print(f"第一个 CSV 字符串比第二个多 {len(remaining_rows1)} 行。")
    elif remaining_rows2:
        has_difference = True
        print(f"第二个 CSV 字符串比第一个多 {len(remaining_rows2)} 行。")

    if not has_difference:
        print("两个 CSV 文件完全一致。")
    return has_difference


if __name__ == '__main__':
    file_names = get_csv_files_without_extension('chunks_data/')
    for filename in file_names:
        try:
            with open(f'chunks_data/{filename}.csv', 'r', encoding='utf-8') as f1:
                file1_str = f1.read()
            with open(f'uncompress_file/{filename}.csv', 'r', encoding='utf-8') as f2:
                file2_str = f2.read()
            compare_csv_files(file1_str, file2_str)
            print(filename)
        except FileNotFoundError:
            print("文件未找到，请检查文件路径。")
        except Exception as e:
            print(f"发生错误: {e}")