import numpy as np
import pandas as pd

# 对数据进行处理并转换为用户-音乐矩阵R
def get_R(filename):
    data=pd.read_csv(filename)
    # 以歌曲播放量占用户总播放量的比例为评分标准
    # 每个用户播放总次数
    all_count = data.groupby('user')['play_count'].sum()
    # 将两个表按user合并
    new_data = pd.merge(data, all_count, on='user')
    new_data['play_count_y']=new_data['play_count_x']/new_data['play_count_y']
    col=['user','song','play_count','score']
    new_data.columns=col
    # 转成矩阵
    R = new_data.pivot_table(index='user', columns='song', values='score')
    R = R.fillna(0)
    return np.array(R.values)

# R是之前的评分矩阵，为PQ分别提供row和col，K是因子数
def init_PQ(R,K):
    P=np.ones([len(R),K])
    Q=np.ones([K,len(R[0])])
    return P,Q

def get_loss(R,P,Q,lamb=0.1):
    # 求当前的predict矩阵
    pred=np.dot(P,Q)
    # 先加上惩罚项
    loss=lamb*np.linalg.norm(P)+lamb*np.linalg.norm(Q)
    for usr in range(len(R)):
        for item in range(len(R[0])):
            loss+=pow(R[usr][item]-pred[usr][item],2)
    return loss

# 学习新的P，Q
def gradDecent(R,P,Q,alpha=0.1,maxIter=100,lamb=0.1,minLoss=0.001):
    K=len(P[0])
    iter=0
    loss = get_loss(R, P, Q, lamb)
    while iter<maxIter and loss>minLoss:
        for usr in range(len(P)):
            for item in range(len(Q[0])):
                eiu=R[usr][item]-np.dot(P,Q)[usr][item]
                for k in range(K):
                    P[usr][k]+=alpha*(eiu*Q[k][item]-lamb*P[usr][k])
                    Q[k][item]+=alpha*(eiu*P[usr][k]-lamb*Q[k][item])
        loss=get_loss(R,P,Q,lamb)
        iter+=1
    print('第',iter,'次迭代')
    return P,Q,loss



R=get_R('user.csv')
P,Q=init_PQ(R,5)
newP,newQ,loss= gradDecent(R,P,Q)
new_R=np.dot(P,Q)



