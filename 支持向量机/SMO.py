import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#  获取特征矩阵和标签向量
# dataMat=[[x11,x12],
#          [x21,x22],
#          ...,
#          [xn1,xn2]]
# labelMat=[-1,-1,1,...,-1,1]
def loadDataSet(fileName):
    dataMat,labelMat = [],[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def drawClassify(dataMat,labelMat):
    positive,negative=[],[]
    for point in range(len(dataMat)):
        if labelMat[point]==1:
            positive.append(dataMat[point])
        else:
            negative.append(dataMat[point])
    return positive,negative

# dataMat,labelMat=loadDataSet('testSet.txt')
# # 可视化数据点
# positive,negative=drawClassify(dataMat,labelMat)
# plt.scatter([point[0] for point in positive],[point[1] for point in positive],label='positive')
# plt.scatter([point[0] for point in negative],[point[1] for point in negative],label='negative')
# plt.legend()
# plt.show()
#####################################################################################################

class parameter:
    """
    参数定义：toler为可容忍的误差或说精度；C为惩罚因子;eCache用于存储Ei，在选择最优j的时候要用到
    """
    def __init__(self,dataMat,labelMat,C,toler):
        self.x = dataMat
        self.y = labelMat
        self.m = len(dataMat)
        self.alpha = np.zeros(self.m)
        self.b = 0
        self.C = C
        self.toler = toler
        self.eCache = np.zeros(self.m)




# 核函数K(x1,x2),分为linear和poly
# 默认为线性
# 线性核函数: K(x1,x2)=x1'x2(内积)
# 多项式核函数: K(x1,x2)=(x1'x2+1)^2
def kernal(x1,x2,type='linear'):
    if type=='linear':
        return sum(x1[i]*x2[i] for i in range(len(x1)))
    else:
        return (sum([x1[i]*x2[i] for i in range(len(x1))]) + 1)**2

# 通过KKT condition的 stationary 可知
# w=∑(alpha[i]*y[i]*x[i])
# f(x)=w'x+b=∑alpha[i]y[i]x[i]'x+b=∑alpha[i]y[i]K(x[i],x)+b
# 此时自变量x是某一个样本点
def f(alpha,x):
    s=os.b
    for i in range(len(os.x)):
        s+=alpha[i]*os.y[i]*kernal(os.x[i],x)
    return s

# 判断KKT条件
# primal问题约束: Y[i]*(w'x+b)-1+sai[i]≥0              sai[i]≥0
# dual问题约束:   alpha[i]≥0                           beta[i]≥0
# 松弛互补:       alpha[i]*(Y[i]*(w'x+b)-1+sai[i])=0   beta[i]sai[i]=0
# 通过KKT condition的 stationary 可知
# alpha=C-beta
# 分类讨论aplha[i]取值
# alpha[i]=0 ==> Y[i]f(X[i])≥1     对应实例点在边界内，即被正确分类的点
# alpha[i]=C ==> Y[i]f(X[i])≤1     在两条边界线之间
# 0<alpha[i]<C ==> Y[i]f(X[i])=1   在边界上，为支持向量
def KKT(alpha,index):
    val=os.y[index]*f(alpha,os.x[index])
    if alpha[index]==0:
        return val>=1
    if alpha[index]==os.C:
        return val<=1
    if 0<alpha[index]<os.C:
        return val==1
    # alpha[i]不在0到C的范围里
    return 0

# E函数
def E(index,alpha):
    return f(alpha,os.x[index])-os.y[index]


def innerL(alpha,i):
    if not KKT(alpha, i):
        # 选择第二个alpha
        E1 = os.eCache[i]
        # 选择|E1-E2|最大时的alpha_j
        maxIdx = -1
        temp = os.eCache[:]
        if E1>0:
            temp[i]=float('inf')
            E2=min(temp)
            maxIdx=np.where(os.eCache==E2)[0][0]
        else:
            temp[i] = float('-inf')
            E2=max(temp)
            maxIdx=np.where(os.eCache==E2)[0][0]
        # 判断上下边界L和H
        L,H=L_H(alpha,i,maxIdx)
        # 若L=H，则alpha必定在边界上，没有优化的空间，可直接返回0值
        if L == H:
            return 0
        # 若eta为0，则返回0，因为分母不能为0，其实eta并不会为负数
        e=eta(i,maxIdx)
        if e == 0:
            return 0
        # 存储旧的alphaj
        alpha_old=alpha[:]
        # 更新并修剪alpha2
        alpha[maxIdx]=update_a2(alpha,i,maxIdx)
        alpha[maxIdx]=clipAlpha(alpha[maxIdx],L,H)
        # 更新alpha1
        alpha[i]=update_a1(alpha,i,maxIdx,alpha[maxIdx])
        # 更新b
        os.b=update_b(alpha,i,maxIdx,alpha_old)
        # 更新eCache
        os.eCache=[E(k,alpha) for k in range(os.m)]
        # 所有参数都更新了，change次数为1
        return 1
    else:
        # 满足KKT，不优化，change次数为0
        return 0



# 计算上下边界L和H
# a1 a2将被限制在[0,C]的box里的线段上
# 4种情况
def L_H(alpha,i,j):
    # k>0的两种情况
    if (os.y[i] != os.y[j]):
        L = max(0, alpha[j] - alpha[i])
        H = min(os.C, os.C + alpha[j] - alpha[i])
    # k<0的两种情况
    else:
        L = max(0, alpha[j] + alpha[i] - os.C)
        H = min(os.C, alpha[j] + alpha[i])
    return L,H


# 修剪新的a2
def clipAlpha(new_a2,L,H):
    if new_a2 > H:
        new_a2 = H
    if L > new_a2:
        new_a2 = L
    return new_a2

# 计算eta=K(x1,x1)+K(x2,x2)-2K(x1,x2)
def eta(i,j):
    return -2*kernal(os.x[i],os.x[j])+kernal(os.x[i],os.x[i])+kernal(os.x[j],os.x[j])

# 更新alphaj, new=old+Y[j](Ei-Ej)/eta
def update_a2(alpha,i,j):
    e=eta(i,j)
    return alpha[j]+os.y[j]*(E(i,alpha)-E(j,alpha))/e

# 更新a1以满足 ∑aiyi=0
# new=old+yiyj(old_a2-new_a2)
# a1和a2满足线性关系
# a1=y1(-∑aiyi(i=3~m)-a2y2)
def update_a1(alpha,i,j,alpha2_new):
    return os.y[i]*(-sum(alpha[k]*os.y[k] for k in range(os.m) if k!=i and k!=j)-alpha2_new*os.y[j])

# 求b
# b1_new=b_old-E1-y1K11(a1_new-a1_old)-y2K21(a2_new-a2_old)
# b2_new=b_old-E2-y1K12(a1_new-a1_old)-y2K22(a2_new-a2_old)
# b_new=平均值
def update_b(alpha,i,j,alpha_old):
    b1,b2=0,0
    if 0<alpha[i]<os.C:
        b1 = os.b - E(i, alpha_old) - os.y[i] * kernal(os.x[i], os.x[i]) * (alpha[i]-alpha_old[i])-os.y[j] * kernal(os.x[i], os.x[j]) * (alpha[j]-alpha_old[j])
    if 0<alpha[j]<os.C:
        b2 = os.b - E(j, alpha_old) - os.y[i] * kernal(os.x[i], os.x[j]) * (alpha[i]-alpha_old[i])-os.y[j] * kernal(os.x[j], os.x[j]) * (alpha[j]-alpha_old[j])
    if b1!=0 and b2!=0:
        return (b1+b2)/2
    else:
        return b1+b2


# 计算w
def cal_w(alpha):
    w = [0,0]
    for i in range(len(os.x)):
        for j in range(len(w)):
            w[j]+=alpha[i]*os.y[i]*os.x[i][j]
    return w

def smo(toler=0.001,maxIters=50):
    # 初始化alpha
    alpha=np.zeros(len(os.x))
    # 统计大循环次数
    iters = 0
    # 用于切换边界、非边界情况
    entireSet = True
    # 统计在边界、非边界情况下是否进行了优化，若当前没有不再有优化则进行切换
    alphaPairsChanged =0
    # 循环结束条件：达到最大迭代次数或迭代无法提高精度(非边界、边界情况下都无法再进行优化)
    while (iters < maxIters) and ((alphaPairsChanged > 0) or (entireSet)):
        # 统计此次循环是否有改变
        alphaPairsChanged = 0
        if entireSet:
            for i in range(len(os.x)):
                alphaPairsChanged+=innerL(alpha,i)
            iters+=1
        else:
            non_list=[i for i in range(len(os.x)) if alpha[i]!=0 and alpha[i]!=os.C]
            for i in non_list:
                alphaPairsChanged+=innerL(alpha,i)
            iters+=1
        # 遍历方式交替
        if entireSet:
            entireSet=False
        # 将非边界情况的迭代结束条件设置为不再有精度提升，这时要考虑边界情况，则再次将entireSet设置为True，利用这种方法进行边界、非边界情况的切换
        elif alphaPairsChanged == 0:
            entireSet = True
    # 得到新的alpha，计算w
    w=cal_w(alpha)
    return w


# 画图
dataMat,labelMat=loadDataSet('testSet.txt')
os=parameter(dataMat,labelMat,1,0.001)
w=smo()
print(w)
# 可视化数据点
positive,negative=drawClassify(dataMat,labelMat)
plt.scatter([point[0] for point in positive],[point[1] for point in positive],label='positive')
plt.scatter([point[0] for point in negative],[point[1] for point in negative],label='negative')
x=np.arange(-2,11)
y=(-w[0]/w[1])*x-os.b/w[1]
plt.plot(x,y)
x=np.arange(-2,11)
y=(-w[0]/w[1])*x-(os.b+1)/w[1]
plt.plot(x,y)
x=np.arange(-2,11)
y=(-w[0]/w[1])*x-(os.b-1)/w[1]
plt.plot(x,y)

plt.legend()
plt.show()





















