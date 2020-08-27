# Deep-Learning-with-PyTorch

#### Image classification by modifying pretrained model
- Classify CIFAR10 Dataset by modifying resnet50/vgg16/densenet121 pretrained model
- The basic difference between the normally implemented model and modifide pretrained model:
```
#import pretrained model
#model = torchvision.models.resnet50(pretrained=True) #uncomment this line for using resnet
model = torchvision.models.vgg16(pretrained=True)  #uncomment this line for using vgg16
#model = torchvision.models.densenet121(pretrained=True) #uncomment this line for using densenet121

#check inside the model architecture
#print(model)

#input_channel = model.classifier.in_features  #uncomment this line for  usingresnet
input_channel = model.classifier[0].in_features  #uncomment this line for using vgg16
#input_channel = model.classifier.in_features  #uncomment this line for using densenet121

#to finetune the model, set requires_grad = False
#OR, comment these below two lines to train entire model
for param in model.parameters():
    param.requires_grad = False
 
#We are going to change only in the fc layer
model.classifier = nn.Sequential(nn.Linear(input_channel, 128),
                         nn.ReLU(),
                         nn.Linear(128, num_classes))


model = model.to(device)

```
