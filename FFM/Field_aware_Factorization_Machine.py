import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split




class FM(object):
    def __init__(self):
        self.data = None
        self.label = None
        self.data_test = None
        self.label_test = None

        self.alpha = 0.01
        self.iter = 30
        self.k= 5
        self._w = None
        self._w_0 = None
        self.v = None

    # 数据归一化，统一标尺
    # 将标签换成-1 1
    def preprocessing(self,DataSet,test_data=False):
        DataSet=np.array(DataSet)
        min_max_scaler = MinMaxScaler()
        data=DataSet[:,:len(DataSet[0])-1]
        label=(DataSet[:,len(DataSet[0])-1:]).T[0]
        for l in range(len(label)):
            if label[l]==0:
                label[l]=-1
        data_minMax = min_max_scaler.fit_transform(data)
        if test_data:
            self.data_test=data_minMax
            self.label_test=label
        else:
            self.data=data_minMax
            self.label=label




# os=FM()
# data_train = pd.read_csv('diabetes_train.txt', header=None)
# data_test = pd.read_csv('diabetes_test.txt', header=None)
#
# os.preprocessing(data_train)
# os.preprocessing(data_test,True)
diabetes = datasets.load_diabetes()
# 442*10的病人-特征矩阵
data=diabetes['data']
label=diabetes['target']

print(len(diabetes['data'][0]))



