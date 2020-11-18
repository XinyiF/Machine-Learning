# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np



def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr=[]
    for row in stringArr:
        temp=[]
        for col in row:
            temp.append(float(col))
        datArr.append(temp)
    return np.array(datArr)
# X:
# [[x00,x01],
# [x10,x11],
# ...]
def PCA(X, K=9999):
    # 去均值化
    meanVals = np.mean(X, axis=0)
    Xt = X - meanVals  # remove mean
    # 求协方差矩阵 C=X*X'/m 注意X是m个列向量矩阵
    # C = np.dot(Xt.T, Xt) / len(Xt)
    C=np.cov(Xt,rowvar=0)
    # 求出协方差矩阵的特征值及对应的特征向量
    # 特征值赋值给a，对应特征向量赋值给b
    a, b = np.linalg.eig(C)
    b=b.T
    # 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 P；
    idx = np.argsort(a)[::-1]
    P, step =[], 0
    for i in idx:
        P.append(b[i])
        if len(P)>=K:
            break
    # 得到降维到K维的数据
    P=np.array(P)
    Y = np.dot(Xt,P.T)
    # 重构原始数据
    X_recon=np.dot(Y,P)+meanVals
    return np.array(Y),np.array(X_recon)

dataMat=loadDataSet('testSet.txt')
lowDMat, reconMat = PCA(dataMat, 1)
plt.scatter(dataMat[:,0], dataMat[:,1])
plt.scatter(reconMat[:,0], reconMat[:,1])
plt.show()

