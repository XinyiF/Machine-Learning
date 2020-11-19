from numpy import linalg as la
import numpy as np


# 评分矩阵
def loadExData():
    return np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

# 计算降维重构后矩阵和原矩阵之间的error
def error(matrix1,matrix2):
    res=0
    for row in range(len(matrix1)):
        for col in range(len(matrix1[0])):
            res+=(matrix1[row][col]-matrix2[row][col])**2
    return np.sqrt(res)

# 向量相似度
# 欧式距离相似度
def ecludSim(A,B):
    return 1.0/(1.0 + la.norm(A - B))

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


def SVDest(rate,userId,itemId):
    U,Sigma,VT=np.linalg.svd(rate)
    idx=np.argsort(Sigma)
    # 特征值排序
    idx=idx[::-1]
    k=enery(sorted(Sigma)[::-1])
    # get切片行列索引
    slice=idx[:k]
    U_new=U[:,slice]
    Sigma_new=np.diag((Sigma)[::-1][:k])
    VT_new=VT[slice,:]
    rate_new=np.dot(U_new,Sigma_new).dot(VT_new)
    # 开始循坏计算其他用户相似度*评分
    score,count_user=0,0
    for row in range(len(rate)):
        if row!=userId:
            sim=pearSim(rate_new[row,:],rate_new[userId,:])
            score+=sim*rate[row][itemId]
            if sim*rate[row][itemId]!=0:
                count_user+=1
    return score/count_user

rate=loadExData()
print(SVDest(rate,4,0))






