import numpy as np
import pandas as pd
import pickle

def mergeFile(filename1,filename2):
    movie=pd.read_csv(filename1)
    rating=pd.read_csv(filename2)
    allData=pd.merge(rating,movie)
    validData=allData.drop(labels='timestamp',axis=1)
    return validData.sort_values(by='userId')

def getGenre(mergedFile):
    # 获取所有流派信息
    genre=[]
    for row in mergedFile.iterrows():
        genList=row[1]['genres'].split('|')
        for gen in genList:
            if gen not in genre:
                genre.append(gen)
    return genre

# 只记录合并的特征矩阵中不为0的index
def featureMatrix(mergedFile,genreName):
    # one-hot编码
    userNum=668
    movieNum=149532
    res,rating=[],[]
    for row in mergedFile.iterrows():
        rate=float(row[1]['rating'])
        rating.append(rate)
        userId=int(row[1]['userId'])-1
        movieId=int(row[1]['movieId'])-1+userNum
        genreId=[]
        for gen in row[1]['genres'].split('|'):
            genreId.append(genreName.index(gen)+userNum+movieNum)
        res.append([userId,movieId]+genreId)
    return res,rating


file=mergeFile('movies.csv','ratings.csv')
genre=getGenre(file)
index,rating=featureMatrix(file,genre)
with open('index.pickle', 'wb') as f:
    pickle.dump(index, f)
    pickle.dump(rating,f)
    pickle.dump(genre,f)










