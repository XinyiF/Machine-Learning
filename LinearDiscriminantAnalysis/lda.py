import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def loadData(trainDataPath):
    data=(pd.read_csv(trainDataPath)).values
    #  提取正负样本
    positiveData,negativeData=[],[]
    for sample in data:
        if sample[-1]==0:
            negativeData.append(sample[:len(sample)-1])
        else:
            positiveData.append(sample[:len(sample)-1])
    return np.array(positiveData),np.array(negativeData)


def lda(positiveData,negativeData):
    # 两类点的概率
    p_pos=len(positiveData)/(len(positiveData)+len(negativeData))
    p_neg=len(negativeData)/(len(positiveData)+len(negativeData))
    # 求两类中心点
    m_pos=np.mean(positiveData,axis=0)
    m_neg=np.mean(negativeData,axis=0)
    # 求Sw(within-class scatter matrix)
    Sw_pos=np.dot((positiveData-m_pos).T,(positiveData-m_pos))
    Sw_neg=np.dot((negativeData-m_neg).T,(negativeData-m_neg))
    Sw=p_pos*Sw_pos+p_neg*Sw_neg

    # 计算e
    e=np.dot(np.mat(Sw).I,(m_pos-m_neg))
    return np.array(e)[0]

def kernal(x1,x2):
    return sum(x1[i]*x2[i] for i in range(len(x1)))

def predict(w,sample,positiveData,negativeData):
    # 求两类中心点
    m_pos=np.mean(positiveData,axis=0)
    m_neg=np.mean(negativeData,axis=0)
    # 求两个类别在w这条线上的投影
    c_pos=kernal(w,m_pos)
    c_neg=kernal(w,m_neg)
    # 两类投影中心
    c=(c_pos+c_neg)/2
    # 该点投影
    c_sample=kernal(w,sample)
    # 分类
    if c_sample-c>=0:
        return 1
    else:
        return 0

def accurancy(w,testSample,positiveData,negativeData):
    res=[]
    for sample in testSample:
        if predict(w,sample,positiveData,negativeData)==sample[-1]:
            res.append(1)
        else:
            res.append(0)
    return sum(res)/len(res)

def draw(w,positiveData,negativeData):
    for pos,neg in zip(positiveData,negativeData):
        dimReduP=kernal(w,pos)
        dimReduN=kernal(w,neg)
        plt.scatter(dimReduP,0,c='r')
        plt.scatter(dimReduN,1, c='g')
    plt.show()







trainDataPath='diabetes_train.txt'
positiveData,negativeData=loadData(trainDataPath)
w=lda(positiveData,negativeData)
testSampe=np.array(pd.read_csv('diabetes_test.txt').values)

print('准确率=',accurancy(w,testSampe,positiveData,negativeData))
draw(w,positiveData,negativeData)
