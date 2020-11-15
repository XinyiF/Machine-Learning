import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import time


# 读取数据，将特征和标签分别存储
def loadData(filename):
    data, dataLabel = [], []
    f = open(filename)
    for line in f.readlines():
        data.append([float(line.split()[0]), float(line.split()[1])])
        dataLabel.append(float(line.split()[2]))
    return data, dataLabel


# sigmoid函数，返回值在0～1之间，可以取一个阈值分类
# z=theta'x
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 内积
def kernal(x1, x2):
    return sum(x1[i] * x2[i] for i in range(len(x1)))


# 代价函数,凸函数
def cost(X, Y, theta):
    c1 = sum(Y[i] * np.log(sigmoid(kernal(theta, X[i]))) for i in range(len(X)))
    c2 = sum((1 - Y[i]) * np.log(1 - sigmoid(kernal(theta, X[i]))) for i in range(len(X)))
    return -(c1 + c2) / len(X)


# 代价函数对theta[0],theta[1],...的偏导
def grad_cost(X, Y, theta):
    res = []
    for j in range(len(theta)):
        res.append(sum((sigmoid(kernal(theta, X[i])) - Y[i]) * X[i][j] for i in range(len(X))) / len(X))
    return res


# 并行计算偏导
def grad_cost_para(X, Y, theta, s=10):
    res = []
    # 把X分为s块，默认为10
    # 前s-1块的size
    size1 = len(X) // s
    # 最后一块的size
    size2 = len(X) - (s - 1) * size1
    for j in range(len(theta)):
        temp = 0
        for idx in range(s):
            if idx == s - 1:
                temp += sum((sigmoid(kernal(theta, X[i])) - Y[i]) * X[i][j] for i in range(len(X)-size2,len(X)))
            else:
                temp += sum((sigmoid(kernal(theta, X[i])) - Y[i]) * X[i][j] for i in range(idx*size1,idx*size1+size1))
        res.append(temp / len(X))
    return res


# 用梯度下降找到合适的theta
# 步长alpha
def grad_decent(X, Y, alpha, init_theta, maxIter=500):
    Iter = 0
    theta = init_theta
    grad_theta = [99, 99]
    while Iter < maxIter and grad_theta != [0, 0]:
        grad_theta = grad_cost(X, Y, theta)
        for i in range(len(theta)):
            theta[i] -= alpha * grad_theta[i]
        Iter += 1
    return theta


def grad_decent_para(X, Y, alpha, init_theta, maxIter=500):
    Iter = 0
    theta = init_theta
    grad_theta = [99, 99]
    while Iter < maxIter and grad_theta != [0, 0]:
        grad_theta = grad_cost_para(X, Y, theta)
        for i in range(len(theta)):
            theta[i] -= alpha * grad_theta[i]
        Iter += 1
    return theta


# 决策边界
# sigmoid(0)=0.5是分类边界
# 0=theta'x
# x[1]=-(theta[0]*x[0])/theta[1]-theta[2]/theta[1]
def drawClassify(dataMat, labelMat):
    positive, negative = [], []
    for point in range(len(dataMat)):
        if labelMat[point] == 1:
            positive.append(dataMat[point])
        else:
            negative.append(dataMat[point])
    return positive, negative


X, Y = loadData('testSet.txt')
# 中心化数据
X = scale(X)
positive, negative = drawClassify(X, Y)
plt.scatter([point[0] for point in positive], [point[1] for point in positive], label='positive')
plt.scatter([point[0] for point in negative], [point[1] for point in negative], label='negative')
t1=time.time()
theta = grad_decent(X, Y, 0.01, [0, 0])
t2=time.time()
print('串行代码用时：',t2-t1)

t1=time.time()
theta1 = grad_decent_para(X, Y, 0.01, [0, 0])
t2=time.time()
print('并行代码用时：',t2-t1)



x = np.arange(-3, 3, 0.1)
y = -(theta[0] * x) / theta[1]
plt.plot(x, y)
plt.legend()
plt.show()
