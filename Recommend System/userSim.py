import numpy as np
from loadData import *

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

# 对原始rating矩阵进行SVD分解
def svd(rating_martix):
    U, Sigma, VT = np.linalg.svd(rating_martix)
    idx=np.argsort(Sigma)
    # 特征值排序
    idx=idx[::-1]
    k=enery(sorted(Sigma)[::-1])
    # get切片行列索引
    slice=idx[:k]
    U_new=U[:,slice]
    Sigma_new=np.diag((Sigma)[::-1][:k])
    VT_new=VT[slice,:]
    # 重构数据矩阵
    rate_new=np.dot(U_new,Sigma_new).dot(VT_new)
    return rate_new


# 向量相似度
# 欧式距离相似度
def ecludSim(A,B):
    return 1.0/(1.0 + np.linalg.norm(A - B))

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


# 相似度矩阵
# 使用重构后的矩阵
def simMatrix(rate_new):
    sim=np.zeros([len(rate_new),len(rate_new)])
    for user1 in range(len(rate_new)):
        for user2 in range(len(rate_new)):
            # 用户和本身相似度设为负无穷
            if user1==user2:
                sim[user1][user2]=float('-inf')
            else:
                sim[user1][user2]=ecludSim(rate_new[user1],rate_new[user2])
    return sim

# 找到和目标用户最相似的K位用户
def closeUser(userID,sim,K):
    simlist=list(sim[userID])
    res=[]
    while len(res)<K and max(simlist)!=float('-inf'):
        maxSim=max(simlist)
        idx=simlist.index(maxSim)
        res.append(idx)
        simlist[idx]=float('-inf')
    return res

# 找到推荐的k部电影
# user: {用户id:[[电影id],[评分]]}
def recomMovie(close_user_ID,user,k):
    all_movie,all_rate=[],[]
    for usr in close_user_ID:
        for movID,rate in zip(user[str(usr)][0],user[str(usr)][1]):
            all_movie.append(int(movID))
            all_rate.append(rate)
    idx=np.argsort(all_rate)[::-1]
    res=[]
    while len(res)<k:
        if all_movie[idx[0]] not in res:
            res.append(all_movie[idx[0]])
            idx=idx[1:]
    return res

def idToname(name,movID):
    res=[]
    for id in movID:
        res.append(name[str(id)])
    return res


def recommend(userID,closeUserNum,recommendMovieNum):
    name, mov_gen, gen_mov = loadGenre('movies.csv')
    rating, user = loadRating('ratings.csv', name)
    user_gen = user_genre(user, mov_gen, gen_mov, name)
    rating_new=svd(user_gen)
    sim=simMatrix(rating_new)
    close_user=closeUser(userID,sim,closeUserNum)
    recommendID=recomMovie(close_user,user,recommendMovieNum)
    return idToname(name,recommendID)
