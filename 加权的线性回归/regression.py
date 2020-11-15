import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
    data,Y =[],[]
    f = open(filename)
    for line in f.readlines():
        temp = []
        line = line.split()
        for i in line[:2]:
            temp.append(float(i))
        Y.append(float(line[2]))
        data.append(temp)
    return data,Y


def theta(x_test,X,Y,k):
    W = []
    for j in range(len(X)):
        W.append(np.exp(abs(x_test - X[j][1]) / (-2 * k)))
    # 构造W矩阵
    W=np.diag(W)
    # 求theta
    X=np.array(X)
    Y = np.array(Y)
    theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
    return theta


# 内积
def kernal(x1, x2):
    return sum(x1[i] * x2[i] for i in range(len(x1)))



X,Y=loadData('ex0.txt')
#k越小拟合程度越高
k=0.03
plt.scatter([X[i][1] for i in range(len(X))],Y)
y=[]
for x in np.arange(0,1,0.01):
    t=(theta(x,X,Y,k))
    y.append(t[0]+x*t[1])

x=np.arange(0,1,0.01)
plt.scatter(x,y,s=2,c='red')
plt.plot()
plt.show()








