# 针对用户相似度的推荐算法
# 这里我将一个用户对于全部流派的评分作为该用户的特征向量，使用KNN算法，对于输入的用户ID
# 首先选出10个与当前用户最相似的10个用户
# 之后选出这10个最相似用户看过的且当前用户没有看过的电影
# 最后在这些电影中选择出 平均评分最高的十部，推荐给用户

from featureAnalysis import *

name,mov_gen,gen_mov=loadGenre('movies.csv')
rating,user=loadRating('ratings.csv',name)
gen_rating=rateGenre(gen_mov,rating)
user_gen=user_genre(user,mov_gen,gen_mov,name)