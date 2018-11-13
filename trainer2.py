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
import vgg
import googlenet
import resnet
import vgg3
import sys



dl = read.Data_Loader(sys.argv[1], 27)

ts, td, tl = dl.create_4d("train")
vs, vd, vl = dl.create_4d("val")

t_dataloader = utils.DataLoader(utils.TensorDataset(td,tl), shuffle=True)
v_dataloader = utils.DataLoader(utils.TensorDataset(vd,vl), shuffle=True)

dataloaders = {"train": t_dataloader, "val":v_dataloader}
dataset_sizes = {"train": ts, "val": vs}
loss_set = {x: [] for x in ['train', 'val']}
acc_set = {x: [] for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()
print ("CUDA: {}".format(use_gpu))


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
                #scheduler.step(metrics=0)
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
                    inputs = Variable(inputs.cuda(async=True))
                    labels = Variable(labels.cuda(async=True))
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


#model = vgg.Convolution_Neural_Network(output_dim=dl.num_classes)
#model = googlenet.GoogLeNet(output_dim=dl.num_classes)
#model = resnet.ResNet(resnet.Bottleneck, [3, 8, 36, 3], num_classes=dl.num_classes)
#model = resnet.ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
model = vgg3.Convolution_Neural_Network(output_dim=dl.num_classes)
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)   
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, cooldown=0)
num_epochs = 40

plt.axis([0, num_epochs, 0, 5])
plt.ion()
plt.xlabel('-- time -->')
plt.ylabel('-- Loss/Acc -->')

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)


x = list(range(num_epochs))
plt.plot(x, loss_set["train"])
plt.plot(x, loss_set["val"])
plt.plot(x, acc_set["train"])
plt.plot(x, acc_set["val"])
plt.legend(['train loss', 'validation loss', 'train accuracy', 'validation accuracy'], loc='upper left')
plt.show()

#torch.save(model.state_dict(), "trained_model/model001/checkpoint.pth.tar")
#torch.save(model, "trained_model/model001/trained_model.dat")
#visualize_model(model)

torch.save(model, 'model_ft.pt')

#model.save_state_dict('trained_model.pth')

while True:
    plt.pause(0.05)
