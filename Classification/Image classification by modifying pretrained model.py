# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:26:35 2020

@author: VRLab_Md. Maklachur Rahman
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform

#configure GPU
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

#set hyper-parameters
batch_sizes = 64
learning_rate = 0.01
epochs = 20
num_classes = 10
num_workers = 0 

#Fetch the dataset for training and testing
train_dataset = torchvision.datasets.CIFAR10(root='Dataset\\',
                                           train=True,
                                           transform=transform.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='Dataset\\',
                                           train=False,
                                           transform=transform.ToTensor(),
                                           download=True)


#use DataLoader to load our dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_sizes,
                                           shuffle=True,
                                           num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_sizes,
                                          shuffle=False,
                                          num_workers=num_workers)

#import pretrained model
model=torchvision.models.resnet50(pretrained=True)

#check inside the model architecture
#print(model)

input_channel = model.fc.in_features
for param in model.parameters():
    param.requires_grad = False
 
#We are going to change only in the fc layer
model.fc = nn.Sequential(nn.Linear(input_channel, 128),
                         nn.ReLU(),
                         nn.Linear(128, num_classes))


model = model.to(device)

# define loss and gradient
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


total_step = len(train_loader)

for epoch in range (epochs):
    #set trining mode 
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print('Epoch: {}/{} [Step: {}/{}] Loss: {:.4f}'.format(epoch+1, epochs, step+1 ,total_step, loss.item()))


#set testing mode            
model.eval()
#test the model
with torch.no_grad():
    total = 0
    correct_predication = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predication = torch.max(outputs, 1)
        total +=labels.size(0)
        correct_predication +=(predication==labels).sum().item()
    print('Accuracy of the model prediction is {:.4f}%'.format(100*correct_predication/total))    


