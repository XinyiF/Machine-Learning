import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 从sklearn自带数据集里取出人脸数据
# 只下载有超过60张照片的人的图像
faces = fetch_lfw_people(min_faces_per_person=60)
# 每张图片像素矩阵
# 可直接imshow
image=faces.images
# 每张图片尺寸62行 47列
row,col=image[0].shape
# X={1347,2914} 一共1288张图像，每张图2914个像素点
X = faces.data
# 每张图对应的人ID
y = faces.target
# 每个ID对应的人名
target_names = faces.target_names
# 选用0.25的数据作为测试集
# 测试照片：337
# 训练照片：1011
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# 使用PCA对训练集降维
n_components = 150
pca=PCA(n_components=n_components, svd_solver='randomized',
          whiten=True, random_state=42).fit(X_train)
# 特征向量方向
# 3d array
# 共150个62 47 矩阵
# 每个矩阵代表一个重要特征
eigenfaces = pca.components_.reshape((n_components,row,col))
# 将训练集和测试集投影到特征向量上
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# 像素矩阵
X_train_img=pca.inverse_transform(X_train_pca)


# 对比图像
fig,(ax1,ax2)=plt.subplots(1,2)
ax1.imshow(X_train[0].reshape(row,col))
ax2.imshow(X_train_img[0].reshape(row,col))
ax1.set_xlabel("full-dim\n input")
ax2.set_xlabel("150-dim\n reconstruction")
# plt.show()




# 通过交叉验证寻找最佳其他参数和kernal形式
# 设置模型可选择的参数范围: C 为模型误分类的惩罚系数   gamma为核函数参数
param_grid = [
    {'kernel': ['linear'], 'C': [1, 5, 10, 50]},
    {'kernel': ['rbf'], 'C': [1, 5, 10, 50], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]},
    {'kernel': ['poly'], 'C': [1, 5, 10, 50], 'degree':[2,3,4], 'gamma': ['auto']}]
# 参数择优模型
# SVC：选择支持向量机模型进行分类
# class_weight = 'balanced'表示样本的权重相等
# cv = 5 表示用五折交叉验证的方法去选择最优参数
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(class_weight='balanced'), param_grid,cv=5,n_jobs = 8)
grid.fit(X_train_pca, y_train)
print(grid.best_estimator_)
print(grid.best_params_)

# 用较好的参数训练
model = grid.best_estimator_
yfit = model.predict(X_test_pca)

# 检查前24个图片识别
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62, 47))
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]],
                   color='black' if yfit[i] == y_test[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
plt.show()

# 整体性能报告
from sklearn.metrics import classification_report
print(classification_report(y_test, yfit,
                            target_names=faces.target_names))

