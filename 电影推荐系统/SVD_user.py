# 建立用户-电影矩阵
# 数据较多且稀疏
# SVD

import csv
import numpy as np
import matplotlib.pyplot as plt
from featureAnalysis import *
from KNN_user import *

# 建立所有用户，所有电影的评分矩阵
# 数据太大不做使用
def loadDate(filename):
    r=open(filename,'r')
    reader=csv.reader(r)
    # 多一行一列使id对应index
    res=np.zeros([669,149533])
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        res[int(item[0])][int(item[1])]=float(item[2])
    return res

def loadRecoDate(filename,recoMovieID):
    r=open(filename,'r')
    reader=csv.reader(r)
    # 多一行用户id对应index
    res=np.zeros([669,len(recoMovieID)])
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        if item[1] in recoMovieID:
            idx=recoMovieID.index(item[1])
            res[int(item[0])][idx]=float(item[2])
    return res


# 向量相似度
# 欧式距离相似度
def ecludSim(A,B):
    return 1.0/(1.0 + np.linalg.norm(A - B))

# 余弦相似度
def cosSim(A,B):
    A_B = sum(A[i] * B[i] for i in range(len(A)))
    cos=A_B/(np.linalg.norm(A)*np.linalg.norm(B))
    return 0.5+0.5*cos

# 皮尔逊相似度（向量减去平均值后做余弦相似度）
def pearSim(A,B):
    meanA=np.mean(A)
    meanB = np.mean(B)
    A_B=sum((A[i]-meanA)*(B[i]-meanB) for i in range(len(A)))
    pear=A_B / (np.linalg.norm(A-meanA) * np.linalg.norm(B-meanB))
    return 0.5+0.5*pear

# 确认SVD应该保留几个特征值90%
# 计算能量
# Sigma升序排列
def enery(Sigma):
    whole_num=len(Sigma)
    Sig2=[]
    for i in Sigma:
        Sig2.append(i**2)
    whole_ener=sum(Sig2)
    for i in range(whole_num):
        if sum(Sig2[0:i])>=0.9*whole_ener:
            return i

# 找到降维后的U sigma VT
def newSVD(U,Sigma,VT,k):
    idx=np.argsort(Sigma)
    slice=idx[:k]
    return U[:,slice],sorted(Sigma)[:k],VT[slice,:]


# 恢复指定用户缺失数据
def svdReco(data,userID):
    # SVD分解
    U,Sigma,VT=np.linalg.svd(data)
    # 找到可以覆盖90%内容的纬度
    k=enery(sorted(Sigma)[::-1])
    U, Sigma, VT=newSVD(U,Sigma,VT,k)
    # 恢复数据
    rate=np.dot(U,np.diag(Sigma)).dot(VT)
    # 缺失行数
    score,count_user=np.zeros(len(data[0])),np.zeros(len(data[0]))
    for row in range(len(data)):
        if row!=userID:
            # 计算相似度
            sim=ecludSim(rate[row],rate[userID])
            for i in range(len(score)):
                score[i]+=sim*data[row][i]
                if data[row][i]!=0:
                    count_user[i]+=1
    for i in range(len(score)):
        if count_user[i]!=0:
            score[i]/=count_user[i]
    return score

def printMovRate(mov_name,rates):
    for i in range(len(mov_name)):
        print('{}  {}{:.2f}'.format(mov_name[i],'predict rate:',rates[i]))





recoMovieID=allRecoMovies('1')
rate=loadRecoDate('ratings.csv',recoMovieID)
res=svdReco(rate,1)
idx=np.argsort(res)[::-1]
# 最好的5部电影
idx=idx[:5]
name, mov_gen, gen_mov = loadGenre('movies.csv')
mov_name=[]
for i in idx:
    mov_name.append(name[recoMovieID[i]])
rates=sorted(res)[::-1][:5]
printMovie(mov_name,mov_gen,rates)




