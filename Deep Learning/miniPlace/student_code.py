# python imports
import os

from torch.onnx.symbolic_opset9 import view
from tqdm import tqdm
import torch.nn.functional as F

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cast = torch.nn.CrossEntropyLoss().to(device)

class SimpleFCNet(nn.Module):
    """
    A simple neural network with fully connected layers
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        super(SimpleFCNet, self).__init__()
        # create the model by adding the layers
        layers = []

        # Add a Flatten layer to convert the 2D pixel array to a 1D vector
        layers.append(nn.Flatten())
        # Add a fully connected / linear layer with 128 nodes
        layers.append(nn.Linear(784,128))
        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))
        # Append a fully connected / linear layer with 64 nodes
        layers.append(nn.Linear(128, 64))
        # Add ReLU activation
        layers.append(nn.ReLU(inplace=True))
        # Append a fully connected / linear layer with num_classes (10) nodes
        layers.append(nn.Linear(64, 10))
        self.layers = nn.Sequential(*layers)
        self.reset_params()

    def forward(self, x):
        # the forward propagation
        out = self.layers(x)
        if self.training:
            # softmax is merged into the loss during training
            # out = nn.functional.log_softmax(out, dim=1)
            return out
        else:
            # attach softmax during inference
            out = nn.functional.softmax(out, dim=1)
            return out

    def reset_params(self):
        # to make our model a faithful replica of the Keras version
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
#######################################################################
class SE(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SE,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels//ratio,channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        #vis.heatmap()
        return x*y

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(2048, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if self.training:
            # softmax is merged into the loss during training
            return out
        else:
            # attach softmax during inference
            out = nn.functional.softmax(out, dim=1)
            return out


    def _make_layers(self):
        cfg=[12,12,'M',64,'M',32,32]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # layers += [nn.Dropout()]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           SE(x, 2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
###################################################################################
# class SimpleConvNet(nn.Module):
#     """
#     A simple convolutional neural network
#     """
#     def __init__(self, input_shape=(32, 32), num_classes=100):
#         super(SimpleConvNet, self).__init__()
#         ####################################################
#         # you can start from here and create a better model
#         ####################################################
#         # this is a simple implementation of LeNet-5
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv1 = nn.Conv2d(3 ,64,1)# 64*32*32
#         # max_pool:64*16*16
#         self.conv2 = nn.Conv2d(64,128,3,stride=3,padding=1)# 128*6*6
#         # max_pool:128*3*3
#         self.flatten=nn.Flatten()
#         self.fc1 = nn.Linear(128*3*3, 1500)
#         self.fc2 = nn.Linear(1500, num_classes)
#
#     def forward(self, x):
#         #################################
#         # Update the code here as needed
#         #################################
#         # the forward propagation
#
#         # attach softmax during inference
#         x = F.relu(self.conv1(x))
#         x=self.pool(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x=self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         if self.training:
#             # softmax is merged into the loss during training
#             return x
#         else:
#             # attach softmax during inference
#             out = nn.functional.softmax(x, dim=1)
#             return out



def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        output=model(input)
        loss = criterion(output,target)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
