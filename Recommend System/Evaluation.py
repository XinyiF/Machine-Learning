from userSim import *
from Evaluation_Index import *
import numpy as np
import random

# 随机挑选testSet
def selectTest(perc_test):
    print('随机挑选测试集...')
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
    return test_user_ID,train_user_ID

test_user_ID,train_user_ID=selectTest(0.01)
# 构造recommends, tests
recommends, tests={},{}
for testId in test_user_ID:
    print('计算第',testId,'位用户...')
    recomName,recomID,user=recommend(int(testId),5,5)
    testMovieID=user[str(testId)][0]
    recommends[testId]=recomID
    tests[testId]=[int(id) for id in testMovieID]
res=precision(recommends, tests)
print('precision is: ',res*100,'%')
rec=recall(recommends, tests)
print('recall is: ',rec*100,'%')





