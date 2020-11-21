import numpy as np
import csv

# 提取用户-电影矩阵
def user_movie_matrix(filename):
    with open(filename,'r') as f:
        reader = csv.reader(f)
        # 668用户，149532部电影（有缺失数据）
        # id对应行列数
        res=np.zeros([669,149533])
        for item in reader:
            # 忽略第一行
            if reader.line_num == 1:
                continue
            res[int(item[0])][int(item[1])]=float(item[2])
        return res


# 记录每位用户观看的电影id
def userWatched(rating_martix):
    res={}
    for user in range(len(rating_martix)):
        res[user] =[]
        for movie in range(len(rating_martix[0])):
            if rating_martix[user][movie]!=0:
                res[user].append(movie)
    return res

# 提取某部电影的流派信息
# name 收录所有电影名称{id:move}
# idx 对应电影ID
# dic 为电影和其对应流派的字典
def loadGenre(filename):
    dm=open(filename,'r')
    reader=csv.reader(dm)
    mov_gen,name,gen_mov={},{},{}
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        name[item[0]]=item[1]
        genre=item[2].split('|')
        mov_gen[item[1]]=genre
        # 将流派和电影对应
        for gen in genre:
            if not gen in gen_mov:
                gen_mov[gen]=[item[1]]
            else:
                gen_mov[gen].append(item[1])
    return name,mov_gen,gen_mov


# 返回{电影名称:[总分，[评分人id]]}
# 返回{用户id:[[电影id],[评分]]}
def loadRating(filename,name):
    dm=open(filename,'r')
    reader=csv.reader(dm)
    rating,user={},{}
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        if not item[0] in user:
            user[item[0]]=[[item[1]],[float(item[2])]]
        else:
            user[item[0]][0].append(item[1])
            user[item[0]][1].append(float(item[2]))

        if not name[item[1]] in rating:
            # [目前总分，[评分人id]]
            rating[name[item[1]]]=[float(item[2]),[item[0]]]
        else:
            rating[name[item[1]]][0]+=float(item[2])
            rating[name[item[1]]][1].append(item[0])
    return rating,user

# 分析各个流派平均分
def rateGenre(genre_mov,rating):
    gen_rating={}
    # 遍历每个流派
    for gen in genre_mov:
        if not gen in gen_rating:
            gen_rating[gen]=0
        # 遍历这个流派下的电影
        score=0
        # 这个流派下电影的数
        num = len(genre_mov[gen])
        for mov in genre_mov[gen]:
            if mov in rating:
                score+=rating[mov][0]/len(rating[mov][1])
        gen_rating[gen]=score/num
    return gen_rating

# 返回每个用户对每个流派的平均评分
# 如没看过则为0分
# {userID:[3,4,2,0,3,...]}
# 序号同genre列表
def user_genre(user,mov_gen,gen_mov,name):
    # genre列表
    gen=[]
    for i in gen_mov:
        gen.append(i)
    user_gen_helper={}
    # user_gen_helper={userID:[[ratings for genre1],[],[],...]}
    for id in user:
        # 初始化
        user_gen_helper[id]=[]
        for i in range(len(gen)):
            user_gen_helper[id].append([])
        # 遍历该用户看过的所有电影
        for mov_idx in range(len(user[id][0])):
            # 这部电影所属流派
            genOfmov=mov_gen[name[user[id][0][mov_idx]]]
            #  这部电影的评分
            score=user[id][1][mov_idx]
            for gen_idx in range(len(gen)):
                if gen[gen_idx] in genOfmov:
                     user_gen_helper[id][gen_idx].append(score)

    # 用户-流派矩阵，用户id对应行数
    user_gen=np.zeros([len(user)+1,len(gen_mov)])
    for id in user_gen_helper:
        for gen_idx in range(len(user_gen_helper[id])):
            if user_gen_helper[id][gen_idx]:
                user_gen[int(id)][gen_idx]=np.mean(user_gen_helper[id][gen_idx])
            else:
                user_gen[int(id)][gen_idx]=0
    return user_gen

