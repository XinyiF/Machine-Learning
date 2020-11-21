import numpy as np


def RMSE(predict,actual):
    """计算均方根误差
        RMSE加大了对预测不准的用户物品评分的惩罚（平方项的惩罚），因而对系统的评测更加苛刻
        @param self.predict, self.actual: 预测评价与真实评价记录的list
        @return: RMSE
    """
    error=sum(pow(pre-act,2) for pre,act in zip(predict,actual))
    return np.sqrt(error/len(predict))

def MSE(predict,actual):
    """计算平均绝对误差
        @param self.predict, self.actual: 预测评价与真实评价记录的list
        @return: MSE
    """
    error=sum(np.abs(pre-act) for pre,act in zip(predict,actual))
    return error/len(predict)

def precision(recommends, tests):
    """
    计算测试集用户的平均精确率：所有被检索到的item中,"应该被检索到"的item占的比例
    :param recommends: 预测出的推荐品字典{ userID : 推荐的物品 }
    :param tests: 真实的观看电影字典{ userID : 实际发生事务的物品 }
    """
    preci=0
    for user in recommends:
        pred=set(recommends[user])
        actual=set(tests[user])
        TP=len(pred & actual)
        preci+=TP/len(pred)
    return preci/len(recommends)

def recall(recommends, tests):
    """
    计算召回率：所有检索到的item占所有"应该检索到的item"的比例
    :param recommends: 预测出的推荐品字典{ userID : 推荐的物品 }
    :param tests: 真实的观看电影字典{ userID : 实际发生事务的物品 }
    """
    reca=0
    for user in recommends:
        pred=set(recommends[user])
        actual=set(tests[user])
        TP=len(pred & actual)
        reca+=TP/len(actual)
    return reca/len(recommends)

def coverage(recommends, all_items):
    """
        计算覆盖率: 推荐系统能够推荐出来的物品占总物品集合的比例
        @param recommends : dict形式 { userID : Items }
        @param all_items :  所有的电影，list
    """
    recom_item=set()
    for user in recommends:
        recom_item=recom_item | (set(recommends[user]) & set(all_items))
    return len(recom_item)/len(all_items)



