# Deep-Learning-with-PyTorch

#### Image classification by modifying pretrained model
Classify CIFAR10 Dataset by modifying Resnet50 pretrained model
The basic difference between the normally implemented model and modifide pretrained model:
'''
#import pretrained model
model=torchvision.models.resnet50(pretrained=True)

#check inside the model architecture
print(model)

input_channel = model.fc.in_features
for param in model.parameters():
    param.requires_grad = False
 
#Change the fc layer
model.fc = nn.Sequential(nn.Linear(input_channel, 128),
                         nn.ReLU(),
                         nn.Linear(128, num_classes))


model = model.to(device) 

'''
