import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib


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
            user_gen_helper[id].append([0])
        # 遍历该用户看过的所有电影
        for mov_idx in range(len(user[id][0])):
            # 这部电影所属流派
            genOfmov=mov_gen[name[user[id][0][mov_idx]]]
            #  这部电影的评分
            score=user[id][1][mov_idx]
            for gen_idx in range(len(gen)):
                if gen[gen_idx] in genOfmov:
                    user_gen_helper[id][gen_idx].append(score)
    user_gen={}
    for id in user_gen_helper:
        user_gen[id]=[]
        for gen_idx in user_gen_helper[id]:
            user_gen[id].append(np.mean(gen_idx))
    return user_gen




def main():
    name,mov_gen,gen_mov=loadGenre('movies.csv')
    rating,user=loadRating('ratings.csv',name)
    gen_rating=rateGenre(gen_mov,rating)
    user_gen=user_genre(user,mov_gen,gen_mov,name)


    # 绘制每种流派的平均得分条形图
    label_list,num_list =[],[]
    for gen in gen_rating:
        label_list.append(gen)
        num_list.append(gen_rating[gen])


    # 各流派平均得分条形图
    params = {'figure.figsize': '25, 4'}
    plt.rcParams.update(params)
    plt.bar(range(len(num_list)), num_list,tick_label=label_list,width=0.5)
    plt.xlabel('Genre')
    plt.ylabel('Rating')
    plt.show()

if __name__ == "__main__":
    main()

