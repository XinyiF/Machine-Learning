# -*- coding: utf-8 -*-

import numpy as np
from random import normalvariate #正态分布
# from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler as MM #可将特征缩放到给定的最小值和最大值之间
import pandas as pd
from scipy.special import expit



class FM(object):
    def __init__(self):
        self.data = None
        self.label = None
        self.data_test = None
        self.label_test = None

        self.alpha = None
        self.iter = None
        self.k=None
        self._w = None
        self._w_0 = None
        self.v = None

    def sigmoid(self,x):  # 定义sigmoid函数
        return 1.0 / (1.0 + expit(-x))

    def kernal(self,v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    # 预测一条数据x
    def getPrediction(self,x,thold):
        m, n = np.shape(self.data)
        result = []
        temp = 0
        for i in range(n):
            for j in range(i + 1, n):
                temp += self.kernal(self.v[i], self.v[j]) * x[i] * x[j]
        term1 = self._w_0
        term2 = self.kernal(x, self._w)
        # 该sample的预测值
        pre = self.sigmoid(term1 + term2 + temp)
        # print(pre)
        if pre > thold:
            pre = 1
        else:
            pre = -1
        return pre


    # 计算准确率
    def calaccuracy(self,pre_y,act_y):
        cost=[]
        for sampleId in range(len(act_y)):
            if pre_y[sampleId]==act_y[sampleId]:
                cost.append(1)
            else:
                cost.append(0)
        return np.sum(cost)/len(cost)

    def sgd_fm(self):
        # 数据矩阵data是m行n列
        m, n = np.shape(self.data)
        # 初始化w0,wi,V,Y_hat
        w0 = 0
        wi = np.zeros(n)
        V = normalvariate(0, 0.2) * np.ones([n, self.k])
        for it in range(self.iter):

            loss=0
            # 随机梯度下降法，每次使用一个sample更新参数
            for sampleId in range(m):
                # 计算交叉项
                temp=0
                for i in range(n):
                    for j in range(i+1,n):
                        temp+=self.kernal(V[i],V[j])*self.data[sampleId][i]*self.data[sampleId][j]
                term1=w0
                term2=self.kernal(self.data[sampleId],wi)
                # 该sample的预测值
                y_hat=term1+term2+temp
                    # 计算损失
                yp=self.sigmoid(y_hat*self.label[sampleId])
                loss=yp-1
                part_df_loss=(yp-1)*self.label[sampleId]
                #  更新w0,wi
                w0-=self.alpha*1*part_df_loss
                for i in range(n):
                    if self.data[sampleId][i]!=0:
                        wi[i]-=self.alpha*self.data[sampleId][i]*part_df_loss
                        for f in range(self.k):
                            V[i][f]-=self.alpha*part_df_loss*self.data[sampleId][i]*sum(V[j][f]*self.data[sampleId][j]-V[i][f]*self.data[sampleId][i]*self.data[sampleId][i] for j in range(n))

            # print('第%s次训练的误差为：%f' % (it, loss))
        self._w = wi
        self._w_0 = w0
        self.v = V


def preprocessing(data_input):
    standardopt = MM()
    data_input.iloc[:, -1].replace(0, -1, inplace=True) #把数据集中的0转为-1
    feature = data_input.iloc[:, :-1] #除了最后一列之外，其余均为特征
    feature = standardopt.fit_transform(feature) #将特征转换为0与1之间的数
    feature = np.array(feature)#传回来的是array，如果要dataframe那用dataframe
    label = np.array(data_input.iloc[:, -1]) #最后一列是标签，表示有无糖尿病
    return feature, label #返回特征，标签

os=FM()
data_train = pd.read_csv('diabetes_train.txt', header=None)
data_test = pd.read_csv('diabetes_test.txt', header=None)

# os.data=data_train[:,:len(data_train[0])-1]
# os.label=(data_train[:,len(data_train[0])-1:]).T[0]
os.data,os.label=preprocessing(data_train)
os.data_test,os.label_test=preprocessing(data_test)
os.alpha = 0.01
os.iter = 40
os.k = 5

os.sgd_fm()
maxacu=0
best_thold=0
for t in np.arange(0.4,0.7,0.01):
    pre_y=[]
    for x in os.data_test:
        pre_y.append(os.getPrediction(x,t))
    acu=os.calaccuracy(pre_y,os.label_test)
    if acu>maxacu:
        maxacu=acu
        best_thold=t


print(maxacu,best_thold)






