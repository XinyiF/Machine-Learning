import pickle
import numpy as np
from random import normalvariate #正态分布
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split



class FM(object):
    def __init__(self):
        self.data = None
        self.m=None
        self.n=None
        self.label = None
        self.data_test = None
        self.label_test = None

        self.alpha = 0.01
        self.iter = 30
        self.k= 5
        self._w = None
        self._w_0 = None
        self.v = None

    # 数据归一化，统一标尺
    # 将标签换成-1 1
    def preprocessing(self,DataSet,test_data=False):
        DataSet=np.array(DataSet)
        min_max_scaler = MinMaxScaler()
        data=DataSet[:,:len(DataSet[0])-1]
        label=(DataSet[:,len(DataSet[0])-1:]).T[0]
        for l in range(len(label)):
            if label[l]==0:
                label[l]=-1
        data_minMax = min_max_scaler.fit_transform(data)
        if test_data:
            self.data_test=data_minMax
            self.label_test=label
        else:
            self.data=data_minMax
            self.label=label



    def sigmoid(self,x):  # 定义sigmoid函数
        return 1.0 / (1.0 + expit(-x))

    def kernal(self,v1,v2):
        return sum(v1[i]*v2[i] for i in range(len(v1)))

    # 预测一条数据x
    def getPrediction(self,x,thold):
        m, n = len(self.data),2
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
        m, n = self.m,self.n
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
                        if i in self.data[sampleId] and j in self.data[sampleId]:
                            temp+=self.kernal(V[i],V[j])
                term1=w0
                idx_not_zero=self.data[sampleId]
                term2=sum(wi[i] for i in idx_not_zero)
                # 该sample的预测值
                y_hat=term1+term2+temp
                # 计算损失
                loss=(y_hat-self.label[sampleId])**2
                part_df_loss=2*(y_hat-self.label[sampleId])
                #  更新w0,wi
                w0-=self.alpha*part_df_loss
                for i in range(n):
                    if i in self.data[sampleId]:
                        wi[i]-=self.alpha
                        for f in range(self.k):
                            s=0
                            for j in range(n):
                                if j in self.data[sampleId]:
                                    s+=V[j][f]-V[i][f]
                            V[i][f]-=self.alpha*part_df_loss*s

            print('第%s次训练的误差为：%f' % (it, loss))
        self._w = wi
        self._w_0 = w0
        self.v = V






with open('index.pickle', 'rb') as f:
    index=pickle.load(f)
    rating=pickle.load(f)
    genre=pickle.load(f)
X_train, X_test, y_train, y_test = train_test_split(index, rating, test_size=0.33, random_state=42)
os=FM()
os.data=X_train
os.label=y_train
os.m=len(X_train)
os.n=150220

os.sgd_fm()

