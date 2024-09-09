# hw_compress_demo

## 一、问题说明

1. 此代码按照每一列高频次数据的分布情况，分别对于每一列做字典压缩或者优化的字典压缩处理。对于只有单一数值的数据列，只保存数值和该数值出现的次数；对于高频次数据集中在少数数值的数据列采用字典压缩方法，并且尽可能利用码字的表达方式节省数据列编码后的存储空间；对于数值分散且分别均匀的数据列，尽可能通过一些初步转型手段，体现出高频次数据集中在少数数值的统计分布特征，找到有利于字典压缩的数据列表达形式。

2. 所有数据列经过统计、去重或转型处理后拼接成为预处理过的数据列集合，按照每一列的字典编码存入Parquet。

3. 结果数据如下：
   
   - city0-4G-1M
   
   - CSV 文件大小: 814752 字节
   
   - Parquet 文件大小: 243754 字节
   
   - Parquet压缩率: 0.29917570009033423
   
   - 压缩后 Parquet 文件大小: 15174 字节
   
   - 压缩后压到Parquet压缩率: 0.01862407211028632
   
   - 压缩后字典json大小: 5447字节

## 二、程序文件结构说明

- 文件结构图如下：

- ```
  C:.
  |   exact.py
  |   README.md
  |   segment.py
  |   uncompress.py
  |
  +---city0
  +---data
  +---json
  +---orignal_data
  \---__pycache__
          segment.cpython-37.pyc
  ```

- data文件夹存原始的parquet文件

- json文件夹存不同压缩产生的中间数据，有三个字段compress_type，dict和value，compress_type记录压缩方式，dict记录字典压缩后的字典，value记录只有单一元素的列的元素值

- exact.py是压缩代码，程序主入口，segment.py是切分大整数的代码，uncompress.py是解压缩程序入口

- city0-4G-1M-preprocessed.csv是产生的预处理文件

- city0-4G-1M-compress.parquet为处理后存入parquet的文件

- city0-4G-1M-compress.csv为处理后生成的csv文件

- city0-4G-1M-uncompress为解压缩后生成的文件

- exact为程序运行代码 程序运行环境为Python 3.7.7
