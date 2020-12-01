import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 读取数据，将特征和标签分别存储
def loadData(filename):
    f = np.array(pd.read_csv(filename).values)
    data,dataLabel=f[:,:len(f[0])-1],f[:,-1]
    data=scale(data)
    data=preprocessing(data)
    return data, dataLabel


# sigmoid函数，返回值在0～1之间，可以取一个阈值分类
# z=theta'x
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 内积
def kernal(x1, x2):
    return sum(x1[i] * x2[i] for i in range(len(x1)))


# 代价函数,凸函数
# 单个样本
def cost(X, Y, theta):
    c1 = Y * np.log(sigmoid(kernal(theta, X)))
    c2 = (1 - Y) * np.log(1 - sigmoid(kernal(theta, X)))
    return c1+c2


# 代价函数对theta[0],theta[1],...的偏导
# 对所有sample
def grad_cost(X, Y, theta):
    res = []
    for j in range(len(theta)):
        res.append(sum((sigmoid(kernal(theta, X[i])) - Y[i]) * X[i][j] for i in range(len(X))) / len(X))
    return res



# 用梯度下降找到合适的theta
# 步长alpha
def grad_decent(X, Y, alpha, maxIter=100):
    Iter = 0
    theta = np.random.random(len(X[0]))
    while Iter < maxIter:
        grad_theta = grad_cost(X, Y, theta)
        for i in range(len(theta)):
            theta[i] -= alpha * grad_theta[i]
        Iter += 1
    return theta


def predict(theta,x,thold):
    pred=kernal(x,theta)
    pred=sigmoid(pred)
    if pred>=thold:
        return 1
    else:
        return 0

# 数据归一化，统一标尺
def preprocessing(data):
    min_max_scaler = MinMaxScaler()
    data_minMax = min_max_scaler.fit_transform(data)
    return data_minMax



X, Y = loadData('diabetes_train.txt')
theta=grad_decent(X,Y,0.01)
X_test,Y_test=loadData('diabetes_test.txt')
maxAcu,bestT=0,0
for t in np.arange(0.1,0.9,0.01):
    acu = []
    for x in range(len(X_test)):
        if predict(theta,X_test[x],t)==Y_test[x]:
            acu.append(1)
        else:
            acu.append(0)
    if sum(acu)/len(acu)>maxAcu:
        maxAcu=sum(acu)/len(acu)
        bestT=t
print('准确率:',maxAcu*100,'%','最佳阈值:',bestT)



