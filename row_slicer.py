import csv
import os
from typing import List, Optional, Union
import glob


def get_csv_files_without_extension(directory):
    # 使用glob模块查找目录下所有.csv文件，然后去掉扩展名
    return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f'{directory}/*.csv')]

class CSVChunkMaster:
    """CSV文件分块处理工具"""
    
    def __init__(
        self,
        rows_per_chunk: int = 5000,
        header: bool = True,
        encoding: str = 'utf-8'
    ):
        """
        初始化CSV分块处理器
        
        参数:
            rows_per_chunk: 每个分块的行数
            header: 是否包含表头
            encoding: 文件编码
        """
        self.rows_per_chunk = rows_per_chunk
        self.header = header
        self.encoding = encoding
    
    def process_file(
        self,
        input_file: str,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        处理单个CSV文件并生成分块
        
        参数:
            input_file: 输入文件路径
            output_dir: 输出目录，默认为原文件所在目录
            
        返回:
            生成的分块文件路径列表
        """
        # 验证输入文件
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
            
        if not input_file.lower().endswith('.csv'):
            raise ValueError(f"文件不是CSV格式: {input_file}")
        
        # 准备输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.dirname(input_file)
        
        # 提取文件名信息
        base_name = os.path.basename(input_file)
        file_stem, file_ext = os.path.splitext(base_name)
        
        # 分块处理
        output_files = []
        current_chunk = []
        chunk_number = 1
        
        try:
            with open(input_file, 'r', newline='', encoding=self.encoding) as infile:
                reader = csv.reader(infile)
                
                # 读取表头
                header_row = next(reader) if self.header else None
                
                for row in reader:
                    current_chunk.append(row)
                    
                    if len(current_chunk) >= self.rows_per_chunk:
                        output_file = self._write_chunk(
                            output_dir, file_stem, file_ext, 
                            chunk_number, current_chunk, 
                            header_row
                        )
                        output_files.append(output_file)
                        current_chunk = []
                        chunk_number += 1
                
                # 处理剩余数据
                if current_chunk:
                    output_file = self._write_chunk(
                        output_dir, file_stem, file_ext, 
                        chunk_number, current_chunk, 
                        header_row
                    )
                    output_files.append(output_file)
            
            return output_files
            
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            return []
    
    def _write_chunk(
        self,
        output_dir: str,
        file_stem: str,
        file_ext: str,
        chunk_number: int,
        rows: List[List[str]],
        header_row: Optional[List[str]] = None
    ) -> str:
        """写入单个分块文件"""
        output_file = os.path.join(output_dir, f"{file_stem}_{chunk_number}{file_ext}")
        
        with open(output_file, 'w', newline='', encoding=self.encoding) as outfile:
            writer = csv.writer(outfile)
            
            if header_row is not None:
                writer.writerow(header_row)
                
            writer.writerows(rows)
        
        return output_file

# 使用示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = CSVChunkMaster(
        rows_per_chunk=5000,
        header=True,
        encoding='utf-8'
    )
    file_names = get_csv_files_without_extension('compress_data/csv/')
    for filename in file_names:
        print(filename)
    # 处理单个文件
        try:
            output_files = processor.process_file(
                input_file=f"compress_data/csv/{filename}.csv",
                output_dir="chunks_data"
            )
            print(f"成功生成 {len(output_files)} 个分块文件")
        except Exception as e:
            print(f"错误: {str(e)}")