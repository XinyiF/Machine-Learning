from featureAnalysis import *
from KNN_user import *
from Evaluation_Index import *
import random
import csv
import numpy as np



def evalu(perc_test,K):
    # 提取某部电影的流派信息
    # name 收录所有电影名称{id:movie}
    # idx 对应电影ID
    # dic 为电影和其对应流派的字典
    name, mov_gen, gen_mov = loadGenre('movies.csv')
    # 返回{电影名称:[总分，[评分人id]]}
    # 返回{用户id:[[电影id],[评分]]}
    rating, user = loadRating('ratings.csv', name)
    # 确认训练集和测试集包含用户人数
    train_user_num=int(len(user)*(1-perc_test))
    test_user_num=len(user)-train_user_num
    # 随机挑选测试集
    test_user_ID=[]
    while len(test_user_ID)<test_user_num:
        id=str(random.randint(0,len(user)-1))
        if not id in test_user_ID:
            test_user_ID.append(id)
    user_test={}
    for id in test_user_ID:
        user_test[id]=user[id]
    # 生成训练集
    train_user_ID=[id for id in user if not id in test_user_ID]
    user_train={}
    for id in train_user_ID:
        user_train[id]=user[id]
    # 训练集用户对各个流派平均评分
    user_gen = user_genre(user_train, mov_gen, gen_mov, name)
    # 预测测试集用户
    recommend,test={},{}
    for test_id in test_user_ID:
        #  得到推荐电影
        dis_mat = dis_matrix(user_gen)
        close_user = closeUser(test_id, dis_mat,K)
        print(close_user)

evalu(0.2,4)


