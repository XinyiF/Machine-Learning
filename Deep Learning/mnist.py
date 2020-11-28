from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # sigmoid函数，返回值在0～1之间，可以取一个阈值分类
    # z=theta'x
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    # sigmoid 函数求导
    def deriv_sigmoid(self,x):
        # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    # input*wight+bias
    # 激活函数用sigmoid
    def feedforward(self, inputs):
        total=inputs
        for w,b in zip(self.weights,self.bias):
            total = np.dot(total,w) + b
            total=self.sigmoid(total)
        return total

    # 训练
    # 损失函数MSE
    def loss(self,predict, actual):
        return sum(pow(predict[i]-actual[i],2) for i in range(len(predict)))/len(predict)

    def deriv(self):
        pass


mnist = keras.datasets.mnist
# 每张图片都是28*28
# 训练集60000张图片
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 压平
train_images=train_images.reshape(60000,784)
# 数据归一化
min_max_scaler = MinMaxScaler()
train_images=min_max_scaler.fit_transform(train_images)

# 设置两个隐藏层，设置神经元个数
layer1=50
layer2=100
# 初始化W1,W2,W3
W1=np.random.randn(784,layer1)
b1=np.random.randn(1,layer1)
W2=np.random.randn(layer1,layer2)
b2=np.random.randn(1,layer2)
W3=np.random.randn(layer2,10)
b3=np.random.randn(1,10)
W=[list(W1),list(W2),list(W3)]
B=[list(b1),list(b2),list(b3)]

ne=Neuron(W,B)
pre=ne.feedforward(train_images[0])
print(pre)
act=[0,0,0,0,0,1,0,0,0,0]
l=ne.loss(pre[0],act)


















#
#
#

# pred=predict(W,train_images[0])



#
# def numerical_gradient(f,x):
#     h=1e-4
#     grad=np.zeros_like(x)
#
#     for idx in range(len(x)):
#         tep_val=x[idx]
#         x[idx]=tep_val+h
#         fxh1=f(x)
#
#         x[idx]=tep_val-h
#         fxh2=f(x)
#
#         grad[idx]=(fxh1-fxh2)/(2*h)
#         x[idx]=tep_val
#
#     return grad
