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

def find_ner(l) :
    pass

def SegmentFind(l,n) :


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
        if len(x) <= 16 and n - i >= 2:
            choose.append([len(x),len(y) / (10 ** i),i])
    choose.sort(key = lambda x : -x[0])
    return choose[0] if len(choose)>0 else [0,0,0]



def SegmentExecute(df,columnname):
    # print(df)
    #print(df)
    column,n = supply(df)
    #print(df)

    x,y,point = SegmentFind(column,n)
    #column,_ = supply(column)
    #print(column)

    if point>0 :


        c = [[] for i in range(point+1)]
        for i,x in enumerate(column) :

            c[0].append(int(x[:-point]))
            tt = x[-point:]
            for j,y in enumerate(tt) :
                c[j+1].append(int(y))



            #print(c2)
        dfn = pd.DataFrame({columnname+'_'+str(i):c[i] for i in range(len(c))})
        #print(dfn)
    else :
        #print(column)
        c = [[] for i in range(len(column[0]))]
        for i, x in enumerate(column):


            for j, y in enumerate(x):
                c[j].append(int(y))
        dfn = pd.DataFrame({columnname + '_' + str(i): c[i] for i in range(len(c))})
    return dfn


