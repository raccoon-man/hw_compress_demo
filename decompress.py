import csv
import re

import numpy as np
import pandas as pd

import json
import os
import re
import fastparquet


def parquetTOcsv(filename,f) :
# 读取 Parquet 文件
    pf = fastparquet.ParquetFile(filename)
    df = pf.to_pandas()

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(f, index=False)

def parquetTOlist(filename) :
    pf = fastparquet.ParquetFile(filename)
    df = pf.to_pandas()
    #df = df.values.tolist()
    return df

def unparquet(filename) :
    #filename = 'city0-4G-1M-compress.parquet'
    #prename = filename[:16]
    parquetTOcsv(filename, filename[:-7] + 'csv')

def ReadInP(FileName) :
    #将数据读入 转化为二维列表形式
    #FileName += '.csv'
    FileContent = []
    with open(FileName, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # 每个row是一个列表，包含每一行的数据
            #print(row)
            FileContent.append(row)
    #FileContent是内容的二维列表
    return FileContent

def Transpose(l) :
    #转置二维列表
    ans = [[] for i in range(len(l[0]))]
    for i,x in enumerate(l) :
        for j,y in enumerate(x) :
            ans[j].append(y)
    return ans

def uncompressexe(table):
    for i,x in enumerate(table) :
        pass

def creatjsondic(path) :
    dic = {}
    contents = os.listdir(path)
    for i in contents :
        path1 = os.path.join(path,i)
        s = i.split('-')
        dic[s[4]] = jsontodic(path1)
        #print(path1)
    return dic

def columntodata(column,dic) :
    #print(dic)
    ans  = []
    for i,x in enumerate(column) :
        if x in dic :
            ans.append(dic[x])
        else :
            ans.append(x)

    return ans

def tabletodic(table) :
    dic = {}
    for i,x in enumerate(table) :
        #ll = []
        #for xx in x[1:] :
        #    ll.append(int(xx))
        #dic[x[0]] = ll
        dic[x[0]] = x[1:]
    return dic

def union(l) :
    ans = []
    l = Transpose(l)
    for x in l :
        t = ''.join(x)
        if t.isdigit() :
            t = str(int(t))

        if t[0] == ';' :
            t = ''
        ans.append(t)
    return ans

def unioncolumn(namelist,table) :
    #print(namelist)
    name = []
    namemake = []
    for i,x in enumerate(namelist) :
        #print(x)
        t = x.split('_')

        tt = '_'.join([t[0],t[1]])
        #print(tt)
        if len(name)>0 and x != tt  :
            if namemake[-1][0] == tt :
                #print(namemake[-1][0])
                namemake[-1].append(x)
            else :
                #print('.....')
                name.append(tt)
                namemake.append([tt,x])
        else :
            name.append(tt)
            namemake.append([tt])
    #print(name)
    #print(namemake)
    newtable = []
    index = 0
    jndex = 0
    n = len(table)
    for jndex in namemake :
        if len(jndex) == 1 :
            newtable.append(table[index])
            index+=1

        else:
            nn = len(jndex) - 1
            dolist = []
            for i in range(nn) :
                dolist.append(table[index])
                index+=1
            newtable.append(union(dolist))

    return newtable,name

def trandic(dic) :
    new ={}
    for k,v in dic.items() :
        k = str(k)

        v = str(v)
        new[v] = k
    return new

def totable(table,namelist,jsonlist,n) :
    newtable = []
    for _,x in enumerate(namelist) :
        # print(jsonlist)
        dic = jsonlist[x]
        way = dic["compress_type"]
        if way == "single_value":
            ttt = dic['value']
            if ttt == 'nan' :
                ttt = ""
            newtable.append([ttt]*n)
        elif way == "dict" :
            #print(trandic(dic['dict']))
            column = columntodata(table[x],trandic(dic['dict']))
            newtable.append(column)
        else :
            newtable.append(table[x])
        #print(len(newtable[-1]))
    #print(newtable[1])
    return newtable


def WriterToExcel(FileName,outcomplete,) :
    '''with open(FileName, mode='w', newline='', encoding='utf-8') as csvfile:
        #print('8')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(outcomplete)'''
    #print(outcomplete)

    #outcomplete = np.array(outcomplete)
    #outcomplete = np.where(np.isnan(outcomplete), '', outcomplete)
    #print(outcomplete)
    # print(outcomplete)

    np.savetxt(FileName, outcomplete, fmt='%s', delimiter=',', comments='')


    #print("已写入csv", FileName)
    return



def jsontodic(filename) :
    with open(filename, 'r') as file:
        # 使用json.load()函数将文件内容加载为字典
        data = json.load(file)

    # 现在data变量包含了JSON文件中的数据，以字典形式表示
    return data

def check(a,b,namelist) :
    for i,x in enumerate(a) :

        if x!=b[i] :
            print(namelist[i])
            for j,y in enumerate(x) :
                if y!=b[i][j] :
                    #print('--')
                    print(y)
                    #print('**')
                    print(b[i][j])
                    #print(y,b[i][j])
                    break
    return

def trantableint(t,filename,namelist) :
    ans = []
    nt = 'json/' + filename.split('/')[-1][:-17] + '/' + filename.split('/')[-1][:-16] + 'dtypes.json'
    nt = jsontodic(nt)

    for i,x in enumerate(t):
        xx = namelist[i]
        xx = xx.split('_')
        xx = "_".join([xx[0],xx[1]])
        if nt[xx] == 'float64' :
            ans.append(detelenan(x))
            continue
        ans.append(tranint(x))
    return ans

def todo(filename ) :
    csvvpath = filename[:-7]+'csv'
    # 使用os.path.splitext()分离文件名和扩展名
    # 使用os.path.splitext()分离文件名和扩展名
    file_name_with_extension = os.path.basename(filename)
    file_name, _ = os.path.splitext(file_name_with_extension)

    # 移除"-compress"后缀
    extracted_part = file_name.replace('-compress', '')

    parquetTOcsv(filename,csvvpath)
    table = ReadInP(csvvpath)
    #os.remove(csvvpath)
    table = Transpose(table)
    #print(table)
    prename = filename[:12]
    #table = Transpose(table)
    n = len(table[0]) - 1
    table = tabletodic(table)
    #print(table)

    jsonlist = 'json/'+filename.split('/')[-1][:-17]
    #print(jsonlist)
    jsonlist = creatjsondic(jsonlist)
    #print(jsonlist)
    #namelist = 'json/city0-4G-1M-preprocessed-columns_list.json'
    namelist = 'json/'+filename.split('/')[-1][:-17]+'/'+filename.split('/')[-1][:-16]+'preprocessed-columns_list.json'
    #print(namelist)
    namelist = jsontodic(namelist)
    newtable = totable(table, namelist, jsonlist, n)
    #print(newtable)
    newtable = trantableint(newtable,filename,namelist)
    newtable, namelist = unioncolumn(namelist, newtable)
    #check(che, newtable)
    che = f'data/{extracted_part}.csv'
    che = ReadInP(che)[1:]
    che = Transpose(che)
    check(che, newtable,namelist)
    newtable = Transpose(newtable)

    newtable = [namelist] + newtable
    # name = input('请输入解压缩后的名称:')
    name = f'decompression_data/{extracted_part}-decompress.csv'
    print(name)
    WriterToExcel(name, newtable)

def is_float(s):
    # 正则表达式匹配小数
    t = s.split('.')
    if len(t) != 2 :
        return 0
    if  t[1].isdigit() :

        if t[0].isdigit() :
            return 1
        if t[0][0] == '-' and t[0][1:].isdigit() :
            return 1



    return 0

def detelenan(l) :
    ans = []
    for i, x in enumerate(l):
        if x == 'nan':
            ans.append('')
        else:
            ans.append(x)
    return ans

def tranint(l) :
    ans = []
    for i,x in enumerate(l) :
        if is_float(x) :
            ttt = str(int(float(x)))
            ans.append(ttt)
        elif x == 'nan' :
            ans.append('')
        else:
            ans.append(x)
    return ans

def todo2() :
    t1 = "compress_data/parquet/city0-4G-1M-compress.csv"
    t2 = "compress_data/parquet/city0-4G-1M-compress1.csv"
    table1 = ReadInP(t1)
    # os.remove(csvvpath)
    table1 = Transpose(table1)
    tb1 = tabletodic(table1)
    table2 = ReadInP(t2)
    # os.remove(csvvpath)
    table2 = Transpose(table2)
    tb2 = tabletodic(table2)
    dic = {}
    for k,v in tb2.items() :
        if tb1.get(k,-1) == -1 :
            dic[k] = v
    dfn = pd.DataFrame({k:v for k,v in dic.items()})
    dfn.to_csv('output.csv', index=True)


if __name__ == '__main__' :
    '''#filename = 'city0-4G-1M-compress.parquet'
    #prename = filename[:16]
    #unparquet(filename)
    filename = 'city0-4G-1M-compress.csv'
    namelist = 'json/city0-4G-1M-preprocessed-columns_list.json'
    namelist = jsontodic(namelist)
    #print(namelist)
    prename = filename[:12]
    table = ReadInP(filename)

    table = Transpose(table)
    #print(table[23])
    che = 'city0-4G-1M.csv'
    che = ReadInP(che)[1:]
    che = Transpose(che)
    #print(che)
    n = len(table[0]) - 1
    table = tabletodic(table)


    jsonlist = 'json/'
    jsonlist = creatjsondic(jsonlist)
    newtable = totable(table,namelist,jsonlist,n)
    #print(newtable)
    newtable,namelist = unioncolumn(namelist,newtable)
    check(che, newtable)
    newtable = Transpose(newtable)

    newtable = [namelist]+newtable
    name = input('请输入解压缩后的名称:')
    WriterToExcel(name,newtable)'''
    #todo2()

    # 指定要遍历的文件夹路径
    folder_path = 'compress_data/parquet'



    for filename in os.listdir(folder_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(folder_path, filename)
            # 在这里可以对读取到的 Parquet 文件数据进行处理
            print(f'compress_data/parquet/{filename}')
            todo(f'compress_data/parquet/{filename}')
    # todo('compress_data/parquet/city0-4G-1M-16777476-compress.parquet')
    #parquetTOcsv('compress_data/parquet/city0-4G-1M-compress.parquet','compress_data/parquet/city0-4G-1M-compress1.csv')
    #print(is_float('-103.0'))









