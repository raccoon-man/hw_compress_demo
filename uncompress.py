import csv

import numpy as np

import json
import os

import fastparquet


def parquetTOcsv(filename,f) :
# 读取 Parquet 文件
    pf = fastparquet.ParquetFile(filename)
    df = pf.to_pandas()

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(f, index=False)


def unparquet(filename) :
    filename = 'city0-4G-1M-compress.parquet'
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
    contents = os.listdir(jsonlist)
    for i in contents :
        path1 = os.path.join(path,i)
        s = i.split('-')
        dic[s[3]] = jsontodic(path1)
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
    print(newtable[1])
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

def check(a,b) :
    for i,x in enumerate(a) :

        if x!=b[i] :
            print(i)
            for j,y in enumerate(x) :
                if y!=b[i][j] :
                    #print('--')
                    #print(y)
                    #print('**')
                    #print(b[i][j])
                    print(y,b[i][j])
                    break
    return

if __name__ == '__main__' :
    #filename = 'city0-4G-1M-compress.parquet'
    #prename = filename[:16]
    #unparquet(filename)
    filename = 'city0-4G-1M-compress.csv'
    namelist = 'json/city0-4G-1M-preprocessed-columns_list.json'
    namelist = jsontodic(namelist)
    #print(namelist)
    prename = filename[:12]
    table = ReadInP(filename)

    table = Transpose(table)
    print(table[23])
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

    WriterToExcel(prename+'uncompress.csv',newtable)









