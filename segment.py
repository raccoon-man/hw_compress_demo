import pandas as pd
from collections import Counter

def supply(l) :
    # print(l)
    nl = list(map(int, l))
    # print(min(nl))
    nnl = max(nl)
    nnn = len(str(nnl))
    for i, x in enumerate(l):
        if len(x) < nnn:
            l[i] = '0' * (nnn - len(x)) + x
    return l,nnn

def SegmentFind(l) :


    l,n = supply(l)
    choose = []
    for i in range(1,n+1) :
        temp = [[], []]
        for j,y in enumerate(l) :
            temp[0].append(y[:-i])
            temp[1].append(y[-i:])
        #print("切分点位为切掉后"+str(i)+"位的结果:"," 前半段的基数数量为"+str(len(x)),"后半段的基数数量为"+str(len(y)))
        x = Counter(temp[0])
        y = Counter(temp[1])
        #print("切分点位为切掉后" + str(i) + "位的结果:", " 前半段的基数数量为" + str(len(x)),"后半段的基数数量为" + str(len(y)))
        #print(len(y) / (10 ** i))
        if len(x) < 32 and n - i >= 2:
            choose.append([len(x),len(y) / (10 ** i),i])
    choose.sort(key = lambda x : -x[0])
    #print(choose)
    return choose[0]



def SegmentExecute(df,columnname):
    # print(df)
    column = df
    _,_,point = SegmentFind(column)
    #column,_ = supply(column)
    #print(column)
    c1 = []
    c2 = []
    for i,x in enumerate(column) :
        c1.append(x[:-point])
        c2.append(x[-point:])
    #print(c2)
    dfn = pd.DataFrame({columnname+'_0':c1,columnname+"_1":c2})
    #print(dfn)
    return dfn