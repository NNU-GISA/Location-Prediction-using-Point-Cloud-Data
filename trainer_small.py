#from __future__ import print_function, division
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
import read_small as read
import numpy as np
import time
import os
import copy





dl = read.Data_Loader("data2")

ts, td, tl = dl.create_4d("train")
vs, vd, vl = dl.create_4d("val")



t_dataloader = utils.DataLoader(utils.TensorDataset(td,tl))
v_dataloader = utils.DataLoader(utils.TensorDataset(vd,vl))
dataloaders = {"train": t_dataloader, "val":v_dataloader}
dataset_sizes = {"train": ts, "val": vs}
use_gpu = torch.cuda.is_available()
print "CUDA: {}".format(use_gpu)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                print type(inputs)
                print type(labels)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

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
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

class Convolution_Neural_Network(torch.nn.Module):
    def __init__(self, output_dim):
        super(Convolution_Neural_Network, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(4, 10, kernel_size=3))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=4))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(74420, 37210))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        #self.conv.add_module("maxpool_fc1", torch.nn.MaxPool2d(kernel_size=6, stride=2, padding=2))
        self.fc.add_module("fc2", torch.nn.Linear(37210, 3721))
        self.fc.add_module("fc3", torch.nn.Linear(3721, output_dim)) 
    
    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320)
        return self.fc.forward(x)


model = Convolution_Neural_Network(output_dim=dl.num_classes)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 


model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)


visualize_model(model)






