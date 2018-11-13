#memory expensive
import torch
class Convolution_Neural_Network(torch.nn.Module):
    def __init__(self, output_dim):
        super(Convolution_Neural_Network, self).__init__()
        '''
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=10, stride=2, padding=2, dilation=1, groups=1, bias=True)) #126
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=5)) #122

        

        self.conv.add_module("conv_2", torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding=1, dilation=1, groups=1, bias=True)) #115
        self.conv.add_module("relu_2", torch.nn.ReLU())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=5)) #111
        

        self.conv.add_module("conv_3", torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=11, stride=1, padding=1, dilation=1, groups=1, bias=True)) #103
        self.conv.add_module("relu_3", torch.nn.ReLU())
        self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=8)) #96
        

        self.conv.add_module("conv_4", torch.nn.Conv2d(in_channels=48, out_channels=56, kernel_size=10, stride=2, padding=4, dilation=1, groups=1, bias=True)) #48
        self.conv.add_module("relu_4", torch.nn.ReLU())
        self.conv.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=5)) #44
        

        self.conv.add_module("conv_5", torch.nn.Conv2d(in_channels=56, out_channels=64, kernel_size=10, stride=2, padding=2, dilation=1, groups=1, bias=True)) #20
        self.conv.add_module("relu_5", torch.nn.ReLU())
        self.conv.add_module("maxpool_5", torch.nn.MaxPool2d(kernel_size=2, stride=2)) #10
        

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(640, 560))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        #self.conv.add_module("maxpool_fc1", torch.nn.MaxPool2d(kernel_size=6, stride=2, padding=2))
        self.fc.add_module("fc2", torch.nn.Linear(560, 480))
        self.fc.add_module("fc3", torch.nn.Linear(480, 320))
        self.fc.add_module("fc4", torch.nn.Linear(320, 160))
        self.fc.add_module("fc5", torch.nn.Linear(160, output_dim)) 
        '''
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1_1", torch.nn.Conv2d(4, 96, kernel_size=3, groups=2))
        self.conv.add_module("conv_1_2", torch.nn.Conv2d(96, 96, kernel_size=3, groups=2))
        self.conv.add_module("conv_1_3", torch.nn.Conv2d(96, 96, kernel_size=3, groups=2))
        self.conv.add_module("conv_1_4", torch.nn.Conv2d(96, 96, kernel_size=3, groups=2))
        self.bn1 = nn.BatchNorm2d(96)
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        
        self.conv.add_module("conv_2_1", torch.nn.Conv2d(96, 256, kernel_size=3, groups=2, padding=2))
        self.conv.add_module("conv_2_2", torch.nn.Conv2d(256, 256, kernel_size=3, groups=2, padding=2))
        self.conv.add_module("conv_2_3", torch.nn.Conv2d(256, 256, kernel_size=3, groups=2, padding=2))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())
        
        self.conv.add_module("conv_3_1", torch.nn.Conv2d(256, 384, kernel_size=3, groups=2, padding=1))
        self.conv.add_module("conv_3_2", torch.nn.Conv2d(384, 384, kernel_size=3, groups=2, padding=1))
        self.conv.add_module("conv_3_3", torch.nn.Conv2d(384, 384, kernel_size=3, groups=2, padding=1))
        self.bn1 = nn.BatchNorm2d(384)
        #self.conv.add_module("dropout_3", torch.nn.Dropout())
        self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=3))
        self.conv.add_module("relu_3", torch.nn.ReLU())

        self.conv.add_module("conv_4_1", torch.nn.Conv2d(384, 456, kernel_size=3, groups=2))
        self.conv.add_module("conv_4_2", torch.nn.Conv2d(456, 456, kernel_size=3, groups=2))
        self.bn1 = nn.BatchNorm2d(456)
        #self.conv.add_module("dropout_4", torch.nn.Dropout())
        self.conv.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module("relu_4", torch.nn.ReLU())

        self.conv.add_module("conv_5_1", torch.nn.Conv2d(456, 512, kernel_size=3, groups=2, stride=2, padding=2))
        self.conv.add_module("conv_5_2", torch.nn.Conv2d(512, 512, kernel_size=3, groups=2, stride=2, padding=2))
        self.bn1 = nn.BatchNorm2d(512)
        self.conv.add_module("dropout_5", torch.nn.Dropout())
        self.conv.add_module("maxpool_5", torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv.add_module("relu_5", torch.nn.ReLU())

        # self.conv.add_module("conv_6", torch.nn.Conv2d(512, 612, kernel_size=5))
        # self.conv.add_module("dropout_6", torch.nn.Dropout())
        # self.conv.add_module("maxpool_6", torch.nn.MaxPool2d(kernel_size=2))
        # self.conv.add_module("relu_6", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(2048, 2048))
        self.fc.add_module("fcrelu_1", torch.nn.ReLU())
        #self.fc.add_module("fcdropout_1", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(2048, 2048))
        self.fc.add_module("fcrelu_2", torch.nn.ReLU())
        
        self.fc.add_module("fc3", torch.nn.Linear(2048, output_dim))
    
    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 2048)
        return self.fc.forward(x)
