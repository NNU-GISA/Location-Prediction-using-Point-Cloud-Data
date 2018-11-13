
import torch
import torch.nn as nn
class Convolution_Neural_Network(torch.nn.Module):
    def __init__(self, output_dim):
        super(Convolution_Neural_Network, self).__init__()
        
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1_1", torch.nn.Conv2d(4, 25, kernel_size=3))
        self.conv.add_module("conv_1_2", torch.nn.Conv2d(25, 25, kernel_size=3))
        self.conv.add_module("conv_1_3", torch.nn.Conv2d(25, 25, kernel_size=3))
        self.conv.add_module("conv_1_4", torch.nn.Conv2d(25, 25, kernel_size=3))
        self.bn1 = nn.BatchNorm2d(25)
        #self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=3))
        self.conv.add_module("relu_1", torch.nn.ReLU())


        
        self.conv.add_module("conv_2_1", torch.nn.Conv2d(25, 40, kernel_size=3, padding=2))        
        self.conv.add_module("conv_2_2", torch.nn.Conv2d(40, 40, kernel_size=3))
        self.conv.add_module("conv_2_3", torch.nn.Conv2d(40, 40, kernel_size=3))
        self.conv.add_module("conv_2_4", torch.nn.Conv2d(40, 40, kernel_size=3))
        self.conv.add_module("relu_2", torch.nn.ReLU())
        self.bn2 = nn.BatchNorm2d(40)
        #self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=3))
        self.conv.add_module("relu_2", torch.nn.ReLU())
        
        self.conv.add_module("conv_3_1", torch.nn.Conv2d(40, 64, kernel_size=3, padding=1))
        self.conv.add_module("conv_3_2", torch.nn.Conv2d(64, 64, kernel_size=3))
        self.conv.add_module("conv_3_3", torch.nn.Conv2d(64, 64, kernel_size=3))
        self.conv.add_module("conv_3_4", torch.nn.Conv2d(64, 64, kernel_size=3))       
        self.conv.add_module("relu_3", torch.nn.ReLU())
        #self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=3))
        self.bn3 = nn.BatchNorm2d(64)

        self.conv.add_module("conv_4_1", torch.nn.Conv2d(64, 78, kernel_size=3, padding=1))
        self.conv.add_module("conv_4_2", torch.nn.Conv2d(78, 78, kernel_size=3))
        self.conv.add_module("conv_4_3", torch.nn.Conv2d(78, 78, kernel_size=3))
        self.bn4 = nn.BatchNorm2d(78)
        self.conv.add_module("relu_4", torch.nn.ReLU())

        #self.conv.add_module("conv_5_1", torch.nn.Conv2d(62, 84, kernel_size=3, padding=1))
        #self.bn1 = nn.BatchNorm2d(84)

        #self.conv.add_module("conv_6_1", torch.nn.Conv2d(78, 84, kernel_size=3, padding=1))
        #self.bn1 = nn.BatchNorm2d(84)
        #self.conv.add_module("dropout_3", torch.nn.Dropout())
        #self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=3))
        #self.conv.add_module("relu_3", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(1950, 1950))
        self.fc.add_module("fcrelu_1", torch.nn.ReLU())
        self.fc.add_module("fc2", torch.nn.Linear(1950, 1950))
        self.fc.add_module("fc3", torch.nn.Linear(1950, 1950))
        self.fc.add_module("fc4", torch.nn.Linear(1950, output_dim))
    
    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 1950)
        return self.fc.forward(x)
