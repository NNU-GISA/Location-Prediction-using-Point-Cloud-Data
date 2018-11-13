from __future__ import print_function, division
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from skimage import io, transform
from numba import jit
import matplotlib.pyplot as plt
import read
import numpy as np
import time
import os
import copy





dl = read.Data_Loader("data")

ts, td, tl = dl.create_4d("train")
vs, vd, vl = dl.create_4d("val")


t_dataloader = utils.DataLoader(utils.TensorDataset(td,tl))
v_dataloader = utils.DataLoader(utils.TensorDataset(vd,vl))
dataloaders = {"train": t_dataloader, "val":v_dataloader}
dataset_sizes = {"train": ts, "val": vs}
loss_set = {x: [] for x in ['train', 'val']}
acc_set = {x: [] for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()
print ("CUDA: {}".format(use_gpu))


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                del inputs
                del labels
                #torch.cuda.empty_cache()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            loss_set[phase].append(epoch_loss)
            acc_set[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            #print ("Memory Usage: {}MB".format(torch.cuda.memory_allocated()))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(preds[j]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

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
        self.conv.add_module("conv_1", torch.nn.Conv2d(4, 10, kernel_size=5))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        
        self.conv.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())
        
        self.conv.add_module("conv_3", torch.nn.Conv2d(20, 30, kernel_size=5))
        #self.conv.add_module("dropout_3", torch.nn.Dropout())
        self.conv.add_module("maxpool_3", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_3", torch.nn.ReLU())

        self.conv.add_module("conv_4", torch.nn.Conv2d(30, 40, kernel_size=5))
        #self.conv.add_module("dropout_4", torch.nn.Dropout())
        self.conv.add_module("maxpool_4", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_4", torch.nn.ReLU())

        self.conv.add_module("conv_5", torch.nn.Conv2d(40, 50, kernel_size=5))
        #self.conv.add_module("dropout_5", torch.nn.Dropout())
        self.conv.add_module("maxpool_5", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_5", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(800, 200))
        self.fc.add_module("fcrelu_1", torch.nn.ReLU())
        #self.fc.add_module("fcdropout_1", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(200, 100))
        self.fc.add_module("fcrelu_2", torch.nn.ReLU())
        #self.fc.add_module("fcdropout_2", torch.nn.Dropout())
        self.fc.add_module("fc3", torch.nn.Linear(100, 50))
        self.fc.add_module("fcrelu_3", torch.nn.ReLU())

        self.fc.add_module("fc4", torch.nn.Linear(50, 25))
        self.fc.add_module("fcrelu_4", torch.nn.ReLU())

        self.fc.add_module("fc5", torch.nn.Linear(25, output_dim))
    
    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 800)
        return self.fc.forward(x)


model = Convolution_Neural_Network(output_dim=dl.num_classes)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)   
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 
num_epochs = 35

plt.axis([0, num_epochs, 0, 5])
plt.ion()
plt.xlabel('-- time -->')
plt.ylabel('-- Loss/Acc -->')

model = train_model(model, criterion, optimizer, num_epochs=num_epochs)


x = list(range(num_epochs))
plt.plot(x, loss_set["train"])
plt.plot(x, loss_set["val"])
plt.plot(x, acc_set["train"])
plt.plot(x, acc_set["val"])
plt.show()

#torch.save(model.state_dict(), "trained_model/model001/checkpoint.pth.tar")
#torch.save(model, "trained_model/model001/trained_model.dat")
#visualize_model(model)


#model.save_state_dict('trained_model.pt')

while True:
    plt.pause(0.05)
