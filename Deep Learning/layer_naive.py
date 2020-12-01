import numpy as np
class mulLayer:
    # 乘法层
    def __init__(self):
        self.x=None
        self.y = None

    # 向前计算
    def forward(self, x, y):
        self.x,self.y=x,y
        out=self.x*self.y
        return out

    def backward(self, out):
        # 反向计算x，y反转
        dx=out*self.y
        dy=out*self.x
        return dx,dy



class addLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    # 加法层反向计算直接传递数据
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# ReLU层
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # mask是x同shape数组
        # x<=0的位置设为Ture
        # 其余为False
        self.mask = (x <= 0)
        out = x.copy()
        # 将x<=0的参数设为0
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.y=None

    def forward(self,x):
        self.y=1/(1+np.exp(-x))
        return self.y

    def backward(self,dout):
        dx=dout*self.y*(1-self.y)
        return dx

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dx=None

        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self,x):
        out=np.dot(self.W,x)+self.b
        return out

    def backward(self,dout):
        self.dx=np.dot(dout,self.W.T)
        self.db=dout
        self.dW=np.dot(self.x.T,dout)
        return self.dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def cross_entropy_error(self,y, t):
        return -np.sum(np.log(y[i]*t[i] for i in range(len(t))))
